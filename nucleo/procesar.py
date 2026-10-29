"""
De una petición validada a un resultado: adquisición, tensor según la receta del
modelo, inferencia, máscara, capas para el mapa y ZIP.
"""
import base64
import datetime as dt
import time

import numpy as np
from affine import Affine

from . import ambito, config, datos, exportar, inferencia, modelos, recetas
from .trabajos import ErrorConCodigo


def _fecha(txt, campo):
    try:
        return dt.date.fromisoformat(str(txt))
    except ValueError:
        raise ErrorConCodigo(400, f'{campo}: "{txt}" no tiene formato AAAA-MM-DD')


def modelo_por_defecto():
    for n in sorted(modelos.catalogo()):
        if modelos.ficha(n)['estado'] == 'listo':
            return n
    return None


def validar(p):
    """Todo lo que se puede rechazar sin bajar imágenes. Devuelve la petición
    completada (modelo y umbral por defecto) o lanza ErrorConCodigo."""
    p = dict(p)
    cat = modelos.catalogo()
    if not cat:
        raise ErrorConCodigo(503, 'el servidor no tiene ningún modelo en modelos/')
    p['modelo'] = p.get('modelo') or modelo_por_defecto()
    if not p['modelo']:
        estados = {n: modelos.ficha(n)['estado'] for n in cat}
        raise ErrorConCodigo(409, f'ningún modelo está listo para usarse: {estados}')
    if p['modelo'] not in cat:
        raise ErrorConCodigo(400, f'modelo "{p["modelo"]}" no existe. Disponibles: {sorted(cat)}')
    f = modelos.ficha(p['modelo'])
    if f['estado'] == 'sin_entrenar' and not config.PERMITIR_SIN_ENTRENAR:
        raise ErrorConCodigo(409, f'{f["etiqueta"]}: {f["detalle"]}')
    if f['estado'] == 'error':
        raise ErrorConCodigo(503, f'{f["etiqueta"]}: {f["detalle"]}')
    if p.get('umbral') is None:
        p['umbral'] = float(f['umbral'])
    d0, d1 = _fecha(p['desde'], 'desde'), _fecha(p['hasta'], 'hasta')
    if d0 > d1:
        raise ErrorConCodigo(400, 'la fecha inicial es posterior a la final')
    if d1 < dt.date(2017, 3, 28):
        raise ErrorConCodigo(400, 'Sentinel-2 L2A en Planetary Computer empieza en 2017')
    if p['sur'] >= p['norte'] or p['oeste'] >= p['este']:
        raise ErrorConCodigo(400, 'el rectángulo está invertido o es vacío')
    bbox = (p['sur'], p['oeste'], p['norte'], p['este'])
    # el ámbito va antes que el tamaño: un área fuera no mejora por ser pequeña
    permitido, fuera, frac, aviso = ambito.revisar(bbox)
    if not permitido:
        raise ErrorConCodigo(403, aviso)
    p['_ambito'] = {'nombre': config.NOMBRE_AMBITO, 'modo': config.AMBITO,
                    'fraccion_dentro': round(frac, 4), 'fuera_del_ambito': bool(fuera),
                    'aviso': aviso}
    _, ancho, alto = datos.area_km2(bbox)
    lado = max(ancho, alto)
    if lado > config.MAX_LADO_KM:
        raise ErrorConCodigo(413, f'el lado mayor es de {lado:.1f} km y el máximo es '
                                  f'{config.MAX_LADO_KM:.0f} km. Dibuja un área más pequeña.')
    try:
        datos.planificar(bbox, lado_min=int(f['lado_px']))
    except datos.DatosInvalidos as e:
        raise ErrorConCodigo(400, str(e))
    return p


def procesar(p, avance=lambda f, t: None):
    t0 = time.time()
    p = validar(p)
    bbox = (p['sur'], p['oeste'], p['norte'], p['este'])

    def tramo(a, b):
        return lambda f, t: avance(a + (b - a) * f, t)

    avance(0.01, 'Cargando el modelo')
    try:
        sesion, receta = modelos.obtener(p['modelo'])
    except PermissionError as e:
        raise ErrorConCodigo(409, str(e))
    except Exception as e:
        raise ErrorConCodigo(503, f'no se pudo cargar el modelo: {type(e).__name__}: {e}')
    lado = int(receta['lado_px'])
    forma = sesion.get_inputs()[0].shape
    if forma[1] != len(receta['canales']) or forma[2] != lado or forma[3] != lado:
        raise ErrorConCodigo(500, f'el ONNX pide {forma[1:]} y la receta dice '
                                  f'{len(receta["canales"])}x{lado}x{lado}')

    plan = datos.planificar(bbox, lado_min=lado)
    bandas = sorted(set(recetas.bandas_s2(receta)) | {'B02', 'B03', 'B04'})
    pend_min = float(p.get('pendiente_min') or 0)
    con_terreno = recetas.necesita_dem(receta) or pend_min > 0
    try:
        mos = datos.componer(plan, p['desde'], p['hasta'], bandas, con_terreno,
                             nubes_max=float(p.get('nubes_max') or 40),
                             avance=tramo(0.03, 0.62))
    except datos.DatosInvalidos as e:
        raise ErrorConCodigo(422, str(e))
    except Exception as e:
        raise ErrorConCodigo(502, f'fallo al obtener las imágenes: {type(e).__name__}: {e}')

    v_u = datos.recortar(mos['valido'], plan['ventana'])
    cobertura = float(v_u.mean())
    cob_min = float(p.get('cobertura_min') if p.get('cobertura_min') is not None
                    else config.COBERTURA_MIN)
    if cobertura < cob_min:
        raise ErrorConCodigo(422, f'solo el {cobertura*100:.0f} % del área quedó sin nubes '
                                  f'(se exige {cob_min*100:.0f} %). Amplía las fechas o '
                                  'sube la nubosidad máxima.')

    avance(0.63, 'Armando los canales del modelo')
    entrada = {b: mos['bandas'][b] for b in recetas.bandas_s2(receta)}
    tensor = recetas.desde_l2a(entrada, mos['terreno'] if recetas.necesita_dem(receta) else {},
                               mos['valido'], receta)
    prob, n_parches = inferencia.predecir(tensor, mos['valido'], sesion, receta,
                                          avance=tramo(0.65, 0.9))
    del tensor

    avance(0.91, 'Máscara y capas para el mapa')
    f, c, h, w = plan['ventana']
    tr_u = plan['transform'] * Affine.translation(c, f)
    prob_u = datos.recortar(prob, plan['ventana'])
    pend_u = datos.recortar(mos['terreno']['pendiente'], plan['ventana']) if con_terreno else None
    m = inferencia.mascara(prob_u, float(p['umbral']), pend_u, pend_min)
    est = inferencia.estadisticas(prob_u, m)
    epsg = plan['epsg']

    m_g, bounds = exportar.a_grados(m.astype(np.float32), tr_u, epsg, nearest=True)
    p_g, _ = exportar.a_grados(prob_u, tr_u, epsg, nearest=False)
    rgb_g = [exportar.a_grados(datos.recortar(mos['bandas'][b], plan['ventana']), tr_u, epsg,
                               nearest=False)[0] for b in ('B04', 'B03', 'B02')]
    png_m = exportar.png_mascara(m_g)
    png_p = exportar.png_probabilidad(p_g)
    png_rgb = exportar.png_color_natural(*rgb_g)
    gj = exportar.geojson_de_mascara(m, tr_u, epsg)

    ficha = modelos.ficha(p['modelo'])
    manifiesto = {
        'generado': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
        'area': {k: p[k] for k in ('sur', 'oeste', 'norte', 'este')},
        'fechas': [p['desde'], p['hasta']], 'nubosidad_maxima_escena': p.get('nubes_max'),
        'modelo': {k: ficha.get(k) for k in ('nombre', 'etiqueta', 'autor', 'canales',
                                             'lado_px', 'salida', 'sha256',
                                             'entrenamiento', 'verificacion_l4s')},
        'umbral': p['umbral'], 'pendiente_minima_grados': pend_min,
        'adquisicion': {'catalogo': config.STAC, 'coleccion': config.COLECCION_S2,
                        'resolucion_m': config.RESOLUCION_M, 'epsg': epsg,
                        'mascara_nubes': 'SCL, clases ' + str(config.SCL_MALAS),
                        'composicion': 'mediana de las escenas menos nubosas',
                        'escala_l4s': receta.get('escala_bandas') or config.ESCALA_L4S,
                        'modo_falso': config.MODO_FALSO},
        'escenas': mos['escenas'], 'teselas': mos['teselas'],
        'cobertura_sin_nubes': round(cobertura, 4), 'resultado': est,
        'ambito': p['_ambito'],
        'aviso': ('Herramienta de apoyo para priorizar inspección. No es un inventario '
                  'oficial ni sustituye la verificación en campo. Las métricas de '
                  'desempeño del modelo son las de Landslide4Sense, no de esta área.')}
    zipb = exportar.armar_zip([
        ('mascara.tif', exportar.geotiff(m.astype('uint8'), tr_u, epsg, 'uint8',
                                         descripcion='deslizamiento (1) / no (0)')),
        ('probabilidad.tif', exportar.geotiff(np.nan_to_num(prob_u, nan=-1), tr_u, epsg,
                                              'float32', nodata=-1,
                                              descripcion='probabilidad de deslizamiento')),
        ('deteccion.geojson', gj), ('mascara.png', png_m), ('probabilidad.png', png_p),
        ('color_natural.png', png_rgb), ('manifiesto.json', manifiesto)])

    b64 = lambda x: base64.b64encode(x).decode('ascii')
    resultado = {
        'ok': True, 'modelo': p['modelo'], 'etiqueta_modelo': ficha['etiqueta'],
        'umbral': p['umbral'], 'pendiente_min': pend_min, 'bounds': bounds,
        'capas': {'mascara': b64(png_m), 'probabilidad': b64(png_p),
                  'color_natural': b64(png_rgb)},
        'geojson': gj, 'stats': est, 'escenas': mos['escenas'],
        'teselas': mos['teselas'], 'parches': n_parches,
        'cobertura': round(cobertura, 4), 'relleno': plan['relleno'],
        'canales': receta['canales'], 'ambito': p['_ambito'],
        'segundos': round(time.time() - t0, 1), 'modo_falso': config.MODO_FALSO,
    }
    return resultado, {'zip': (zipb, 'application/zip')}
