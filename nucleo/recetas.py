"""
Recetas de entrada de cada modelo.

Un ONNX no dice qué espera en cada canal. Eso va en una receta JSON junto al
.onnx, con el mismo nombre de base:

    modelos/deeplabv3plus.onnx
    modelos/deeplabv3plus.json

La receta fija los canales y su orden, la normalización con sus constantes, el
tipo de salida y el umbral. La función `armar` construye el tensor a partir de
canales ya expresados en unidades de Landslide4Sense, y la usan por igual el
backend (con imágenes de cubo) y la verificación (con los h5 de L4S). Así lo que
se mide contra Landslide4Sense es exactamente lo que corre en producción.
"""
import json
import os

import numpy as np

from . import config

# Posición de cada canal en los h5 de Landslide4Sense (128 x 128 x 14).
IDX_L4S = {'B01': 0, 'B02': 1, 'B03': 2, 'B04': 3, 'B05': 4, 'B06': 5, 'B07': 6,
           'B08': 7, 'B09': 8, 'B10': 9, 'B11': 10, 'B12': 11,
           'pendiente': 12, 'elevacion': 13}
INDICES = {'NDVI': ('B08', 'B04'), 'NDWI': ('B03', 'B08')}
TERRENO = ('pendiente', 'elevacion')
SALIDAS = ('sigmoide', 'softmax')


class RecetaInvalida(ValueError):
    pass


def leer(ruta_json):
    with open(ruta_json, encoding='utf-8') as f:
        r = json.load(f)
    validar(r, os.path.basename(ruta_json))
    return r


def validar(r, origen='receta'):
    for campo in ('nombre', 'onnx', 'canales', 'lado_px', 'normalizacion', 'salida',
                  'umbral'):
        if campo not in r:
            raise RecetaInvalida(f'{origen}: falta "{campo}"')
    for c in r['canales']:
        if c not in IDX_L4S and c not in INDICES:
            raise RecetaInvalida(f'{origen}: canal desconocido "{c}"')
    if r['salida'] not in SALIDAS:
        raise RecetaInvalida(f'{origen}: salida "{r["salida"]}" no es {SALIDAS}')
    n = r['normalizacion']
    if n.get('metodo') not in ('percentil', 'media_desv', 'ninguna'):
        raise RecetaInvalida(f'{origen}: normalización "{n.get("metodo")}" desconocida')
    if n['metodo'] != 'ninguna':
        for c in n.get('canales', []):
            if c not in n.get('por_canal', {}):
                raise RecetaInvalida(f'{origen}: faltan las constantes de "{c}"')
    for c in r['canales']:
        if c in TERRENO and not r.get('escala_terreno', {}).get(c):
            raise RecetaInvalida(
                f'{origen}: el canal "{c}" necesita su constante en escala_terreno, '
                'porque Landslide4Sense no lo trae en metros ni en grados')
    if not 0 < float(r['umbral']) < 1:
        raise RecetaInvalida(f'{origen}: el umbral tiene que estar entre 0 y 1')


def bandas_s2(r):
    """Bandas de Sentinel-2 que hay que pedir para esta receta."""
    s = set()
    for c in r['canales']:
        if c in INDICES:
            s.update(INDICES[c])
        elif c.startswith('B'):
            s.add(c)
    return sorted(s)


def necesita_dem(r):
    return any(c in TERRENO for c in r['canales'])


def _indice(a, b):
    a = a.astype(np.float32); b = b.astype(np.float32)
    den = a + b
    out = np.zeros_like(a)
    np.divide(a - b, den, out=out, where=den != 0)
    return np.clip(out, -1.0, 1.0)


def armar(fuente, valido, r):
    """fuente: canal -> (H, W) en unidades de Landslide4Sense.
    valido: (H, W) bool. Devuelve (C, H, W) float32 listo para el ONNX."""
    n = r['normalizacion']
    a_normalizar = set(n.get('canales', []))
    capas = []
    for c in r['canales']:
        if c in INDICES:
            x = _indice(fuente[INDICES[c][0]], fuente[INDICES[c][1]])
        else:
            x = fuente[c].astype(np.float32)
        if c in a_normalizar:
            k = n['por_canal'][c]
            if n['metodo'] == 'percentil':
                lo, hi = float(k['lo']), float(k['hi'])
                x = np.clip((x - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
            elif n['metodo'] == 'media_desv':
                x = (x - float(k['media'])) / max(float(k['desv']), 1e-9)
        capas.append(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
    t = np.stack(capas).astype(np.float32)
    return t * valido.astype(np.float32)[None]


# ------------------------------------------------------------ dos adaptadores
def desde_l4s(h5, r):
    """Un parche de Landslide4Sense, (128, 128, 14) o (14, 128, 128), ya viene en
    sus unidades. Solo hay que tomar los canales."""
    a = np.asarray(h5, np.float32)
    if a.shape[-1] == 14:
        a = a.transpose(2, 0, 1)
    valido = np.isfinite(a).all(axis=0)
    a = np.nan_to_num(a, nan=0.0)
    necesarios = set()
    for c in r['canales']:
        necesarios.update(INDICES.get(c, (c,)))
    fuente = {c: a[IDX_L4S[c]] for c in necesarios}
    return armar(fuente, valido, r), valido


def desde_l2a(bandas_dn, terreno, valido, r, escala=None):
    """Bandas de Sentinel-2 L2A en niveles digitales (ya sin el desplazamiento de
    la baseline 04.00), y pendiente en grados y elevación en metros. Se llevan a
    unidades de Landslide4Sense dividiendo cada banda por su media de escena,
    que es como se construyó Landslide4Sense."""
    escala = escala or r.get('escala_bandas') or config.ESCALA_L4S
    if not escala:
        raise RecetaInvalida('no hay escala de bandas: falta escala_laconvencion.json')
    fuente = {}
    for b, x in bandas_dn.items():
        if b not in escala:
            raise RecetaInvalida(f'no hay constante de escala para {b}')
        fuente[b] = x.astype(np.float32) / float(escala[b])
    for c, x in (terreno or {}).items():
        k = r.get('escala_terreno', {}).get(c)
        if k:
            fuente[c] = x.astype(np.float32) / float(k)
    return armar(fuente, valido, r)


def probabilidad(salida, tipo):
    """(N, C, H, W) -> (N, H, W) en [0, 1] para la clase deslizamiento."""
    z = np.asarray(salida, np.float32)
    if tipo == 'sigmoide':
        if z.shape[1] != 1:
            raise RecetaInvalida(f'la receta dice sigmoide y el modelo da {z.shape[1]} canales')
        return 1.0 / (1.0 + np.exp(-np.clip(z[:, 0], -60, 60)))
    if z.shape[1] != 2:
        raise RecetaInvalida(f'la receta dice softmax y el modelo da {z.shape[1]} canales')
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e[:, 1] / e.sum(axis=1)
