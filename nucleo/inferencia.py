"""
Inferencia por ventana deslizante.

El área se recorre en parches del tamaño que declara el modelo (128) con solape,
y cada píxel promedia las predicciones de los parches que lo cubren con un peso
de coseno: el centro de un parche pesa más que su borde, que es donde una red
convolucional ve menos contexto. Así no quedan costuras entre parches.
"""
import numpy as np

from . import config, recetas


def _peso(lado):
    v = np.hanning(lado + 2)[1:-1].astype(np.float32)
    return np.maximum(np.outer(v, v), 1e-3)


def _posiciones(n, lado, paso):
    p = list(range(0, max(n - lado, 0) + 1, paso))
    if p[-1] + lado < n:
        p.append(n - lado)
    return p


def predecir(tensor, valido, sesion, receta, avance=None, lote=8):
    C, H, W = tensor.shape
    lado = int(receta['lado_px'])
    if H < lado or W < lado:
        raise ValueError(f'el área ({H}x{W}) es menor que el parche ({lado})')
    paso = max(1, int(round(lado * (1 - config.SOLAPE))))
    peso = _peso(lado)
    suma = np.zeros((H, W), np.float32)
    pesos = np.zeros((H, W), np.float32)
    nombre = sesion.get_inputs()[0].name
    ventanas = [(f, c) for f in _posiciones(H, lado, paso) for c in _posiciones(W, lado, paso)
                if valido[f:f + lado, c:c + lado].mean() >= 0.05]
    for i in range(0, len(ventanas), lote):
        grupo = ventanas[i:i + lote]
        x = np.stack([tensor[:, f:f + lado, c:c + lado] for f, c in grupo])
        p = recetas.probabilidad(sesion.run(None, {nombre: x})[0], receta['salida'])
        for (f, c), pk in zip(grupo, p):
            suma[f:f + lado, c:c + lado] += pk * peso
            pesos[f:f + lado, c:c + lado] += peso
        if avance:
            avance(min(1.0, (i + lote) / max(len(ventanas), 1)),
                   f'Inferencia: parche {min(i + lote, len(ventanas))} de {len(ventanas)}')
    prob = np.full((H, W), np.nan, np.float32)
    ok = pesos > 0
    prob[ok] = suma[ok] / pesos[ok]
    prob[~valido] = np.nan
    return prob, len(ventanas)


def mascara(prob, umbral, pendiente=None, pendiente_min=0.0):
    m = np.nan_to_num(prob, nan=0.0) >= umbral
    if pendiente is not None and pendiente_min > 0:
        m &= np.nan_to_num(pendiente, nan=0.0) >= pendiente_min
    return m


def estadisticas(prob, m):
    v = np.isfinite(prob)
    n_v = int(v.sum())
    ha_px = config.RESOLUCION_M ** 2 / 1e4
    pv = prob[v]
    return {'px_con_dato': n_v, 'px_marcados': int(m.sum()),
            'fraccion_marcada': round(float(m.sum()) / max(n_v, 1), 5),
            'area_marcada_ha': round(float(m.sum()) * ha_px, 2),
            'area_total_ha': round(n_v * ha_px, 2),
            'prob_max': round(float(pv.max()), 4) if n_v else None,
            'prob_mediana': round(float(np.median(pv)), 4) if n_v else None}
