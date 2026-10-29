"""
Carga de modelos ONNX con su receta, y la comprobación de que están entrenados.

Los ONNX que había en el repositorio corrían sin error y no detectaban nada: sus
pesos eran los de inicialización de PyTorch, porque al exportar no se cargó el
checkpoint. Un fallo así no se ve en la forma del grafo ni en un error de
ejecución, solo en los pesos. Por eso cada modelo pasa por `revisar_pesos` antes
de usarse: en una capa recién inicializada, el peso máximo es exactamente
1/sqrt(fan_in), la media de |w| es la mitad y la desviación la de una uniforme.
Una capa entrenada se aleja de eso.
"""
import glob
import hashlib
import json
import os
import threading

import numpy as np

from . import config, recetas

CABECERA_LFS = b'version https://git-lfs'
_LOCK = threading.Lock()
_SESIONES = {}
_FICHAS = {}


# ------------------------------------------------------------------ archivos
def es_puntero_lfs(ruta):
    try:
        with open(ruta, 'rb') as f:
            return f.read(len(CABECERA_LFS)) == CABECERA_LFS
    except OSError:
        return False


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, 'rb') as f:
        for bloque in iter(lambda: f.read(1 << 20), b''):
            h.update(bloque)
    return h.hexdigest()


def _asegurar_archivo(ruta):
    """Si falta o es un puntero de Git LFS, lo baja de URL_PESOS_BASE."""
    if os.path.exists(ruta) and not es_puntero_lfs(ruta):
        return
    if not config.URL_PESOS_BASE:
        raise FileNotFoundError(
            f'{os.path.basename(ruta)} falta o llegó como puntero de Git LFS, y no hay '
            'URL_PESOS_BASE para bajarlo')
    import httpx
    url = f'{config.URL_PESOS_BASE.rstrip("/")}/{os.path.basename(ruta)}'
    print(f'[pesos] bajando {url}', flush=True)
    tmp = ruta + '.parcial'
    with httpx.stream('GET', url, follow_redirects=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, 'wb') as f:
            for trozo in r.iter_bytes(1 << 20):
                f.write(trozo)
    if es_puntero_lfs(tmp):
        os.remove(tmp)
        raise RuntimeError(f'{url} devolvió otro puntero LFS, no el archivo')
    os.replace(tmp, ruta)


# --------------------------------------------------------- prueba de los pesos
def _parece_inicializacion(w):
    """¿Esta capa conv tiene la forma exacta de una inicialización y no de un
    peso entrenado? Se reconocen las de PyTorch por defecto, Kaiming y Xavier,
    uniformes o normales, que son las que usan torchvision y
    segmentation_models_pytorch. Un peso entrenado tiene colas y no calza con
    ninguna a la vez en forma y en escala."""
    w = np.asarray(w, np.float64)
    fi = w.shape[1] * w.shape[2] * w.shape[3]
    fo = w.shape[0] * w.shape[2] * w.shape[3]
    c = np.abs(w).max()
    s = w.std()
    if c == 0 or s == 0:
        return False
    cotas = (1 / np.sqrt(fi), np.sqrt(3 / fi), np.sqrt(6 / fi), np.sqrt(6 / (fi + fo)),
             np.sqrt(3 / fo), np.sqrt(6 / fo))
    uniforme = (0.47 < np.abs(w).mean() / c < 0.53 and 0.95 < s / (c / np.sqrt(3)) < 1.05
                and any(abs(c / k - 1) < 0.03 for k in cotas))
    if uniforme:
        return True
    z = (w - w.mean()) / s
    exceso_curtosis = float((z ** 4).mean() - 3)
    desvs = (np.sqrt(2 / fi), np.sqrt(2 / fo), np.sqrt(1 / fi), np.sqrt(2 / (fi + fo)))
    normal = (abs(w.mean()) < 0.05 * s and abs(exceso_curtosis) < 0.15
              and any(abs(s / k - 1) < 0.02 for k in desvs))
    return bool(normal)


def revisar_pesos(ruta_onnx):
    """Cuántas capas conv tienen pesos indistinguibles de la inicialización
    uniforme de PyTorch, U(-1/sqrt(fan_in), +1/sqrt(fan_in))."""
    import onnx
    from onnx import numpy_helper
    m = onnx.load(ruta_onnx, load_external_data=True)
    pesos = {t.name: t for t in m.graph.initializer}
    marcas = []                      # en el orden del grafo
    for nodo in m.graph.node:
        if nodo.op_type != 'Conv' or len(nodo.input) < 2 or nodo.input[1] not in pesos:
            continue
        t = pesos[nodo.input[1]]
        if len(t.dims) != 4 or int(np.prod(t.dims)) < 256:
            continue
        marcas.append(_parece_inicializacion(numpy_helper.to_array(t)))
    del m, pesos
    convs, en_init = len(marcas), sum(marcas)
    frac = en_init / convs if convs else 0.0
    cabeza = any(marcas[-3:])
    return {'capas_conv': convs, 'capas_en_inicializacion': en_init,
            'fraccion': round(frac, 3), 'cabeza_sin_entrenar': cabeza,
            # la cabeza es la que decide la clase: si está en la inicialización el
            # modelo no aprendió a segmentar, aunque el codificador traiga ImageNet
            'sin_entrenar': frac >= 0.25 or cabeza}


# ----------------------------------------------------------------- catálogo
def catalogo():
    """nombre -> receta, leyendo modelos/*.json."""
    out = {}
    for ruta in sorted(glob.glob(os.path.join(config.CARPETA_MODELOS, '*.json'))):
        try:
            r = recetas.leer(ruta)
        except Exception as e:
            print(f'[modelos] receta ignorada {os.path.basename(ruta)}: {e}', flush=True)
            continue
        r['_ruta_onnx'] = os.path.join(config.CARPETA_MODELOS, r['onnx'])
        out[r['nombre']] = r
    return out


def ficha(nombre):
    """Lo que /modelos publica de cada modelo. Se calcula una vez."""
    if nombre in _FICHAS:
        return _FICHAS[nombre]
    cat = catalogo()
    if nombre not in cat:
        raise KeyError(nombre)
    r = cat[nombre]
    f = {'nombre': nombre, 'etiqueta': r.get('etiqueta', nombre),
         'autor': r.get('autor'), 'canales': r['canales'], 'lado_px': r['lado_px'],
         'salida': r['salida'], 'umbral': r['umbral'],
         'necesita_dem': recetas.necesita_dem(r),
         'entrenamiento': r.get('entrenamiento'),
         'verificacion_l4s': r.get('verificacion_l4s')}
    try:
        _asegurar_archivo(r['_ruta_onnx'])
        ses = _crear_sesion(r['_ruta_onnx'])
        e = ses.get_inputs()[0]
        o = ses.get_outputs()[0]
        f['entrada'] = {'forma': [d if isinstance(d, int) else 'N' for d in e.shape]}
        f['salida_forma'] = [d if isinstance(d, int) else 'N' for d in o.shape]
        c, h, w = e.shape[1], e.shape[2], e.shape[3]
        problemas = []
        if c != len(r['canales']):
            problemas.append(f'el ONNX pide {c} canales y la receta da {len(r["canales"])}')
        if (h, w) != (r['lado_px'], r['lado_px']):
            problemas.append(f'el ONNX pide {h}x{w} y la receta dice {r["lado_px"]}')
        pesos = revisar_pesos(r['_ruta_onnx'])
        f['pesos'] = pesos
        f['sha256'] = sha256(r['_ruta_onnx'])
        if pesos['sin_entrenar']:
            f['estado'] = 'sin_entrenar'
            f['detalle'] = (f'{pesos["capas_en_inicializacion"]} de {pesos["capas_conv"]} '
                            'capas tienen los pesos de inicialización: este ONNX se '
                            'exportó sin cargar el entrenamiento y no detecta nada.')
        elif problemas:
            f['estado'] = 'error'
            f['detalle'] = '; '.join(problemas)
        else:
            f['estado'] = 'listo'
        del ses
    except Exception as e:
        f['estado'] = 'error'
        f['detalle'] = f'{type(e).__name__}: {e}'
    _FICHAS[nombre] = f
    return f


def _crear_sesion(ruta):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = config.HILOS_ONNX
    so.inter_op_num_threads = 1
    so.enable_cpu_mem_arena = False
    return ort.InferenceSession(ruta, so, providers=['CPUExecutionProvider'])


def obtener(nombre):
    """(sesión, receta) lista para inferir, o una excepción que explica por qué no."""
    cat = catalogo()
    if nombre not in cat:
        raise KeyError(f'modelo "{nombre}" no existe. Disponibles: {sorted(cat)}')
    f = ficha(nombre)
    if f['estado'] == 'sin_entrenar' and not config.PERMITIR_SIN_ENTRENAR:
        raise PermissionError(f['detalle'])
    if f['estado'] == 'error':
        raise RuntimeError(f['detalle'])
    with _LOCK:
        if nombre not in _SESIONES:
            if config.UN_MODELO_A_LA_VEZ:
                _SESIONES.clear()
            _SESIONES[nombre] = _crear_sesion(cat[nombre]['_ruta_onnx'])
        return _SESIONES[nombre], cat[nombre]


def cargados():
    return sorted(_SESIONES)
