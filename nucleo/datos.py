"""
Adquisición: del rectángulo del analista a un mosaico de Sentinel-2 L2A sin nubes.

  1. planificar: una malla de 10 m en una sola zona UTM que cubre el rectángulo.
     Si el área tiene menos de 128 px de lado se rellena por fuera hasta 128 y
     al final se recorta, porque el modelo pide parches de 128.
  2. La malla se cubre con cubos cuadrados de cubo, que solo crea cuadrados.
  3. Por tesela: las escenas menos nubosas de la ventana de fechas, el
     desplazamiento de 1000 de la baseline 04.00 restado por escena, la capa SCL
     para quitar nubes y sombras, y la mediana de lo que queda.
  4. Si hace falta relieve: el DEM a sus 30 m nativos con margen alrededor, la
     pendiente calculada ahí y recién después llevada a 10 m. Derivar una
     elevación ya remuestreada a 10 m da una pendiente a escalones, y el borde
     del mosaico, donde la diferencia es de un solo lado, se infla.
"""
import math

import numpy as np
from pyproj import Transformer

from . import config


class DatosInvalidos(ValueError):
    pass


# --------------------------------------------------------------- geometría
def epsg_utm(lat, lon):
    zona = int((lon + 180) // 6) + 1
    return (32700 if lat < 0 else 32600) + zona


def area_km2(bbox):
    sur, oeste, norte, este = bbox
    lat = math.radians((sur + norte) / 2)
    ancho = (este - oeste) * 111.320 * math.cos(lat)
    alto = (norte - sur) * 110.574
    return ancho * alto, ancho, alto


def planificar(bbox, lado_min=128, lado_tesela=None):
    sur, oeste, norte, este = bbox
    zonas = {epsg_utm(la, lo) for la in (sur, norte) for lo in (oeste, este)}
    if len(zonas) > 1:
        raise DatosInvalidos(
            'el rectángulo cruza el meridiano 72° O, que separa dos zonas UTM. '
            'Muévelo un poco a un lado; en La Convención solo una franja del extremo '
            'este queda en la zona 19.')
    epsg = zonas.pop()
    a_utm = Transformer.from_crs(4326, epsg, always_xy=True)
    xs, ys = a_utm.transform([oeste, este, oeste, este], [sur, sur, norte, norte])
    r = config.RESOLUCION_M
    x0 = math.floor(min(xs) / r) * r
    x1 = math.ceil(max(xs) / r) * r
    y0 = math.floor(min(ys) / r) * r
    y1 = math.ceil(max(ys) / r) * r
    ancho, alto = int(round((x1 - x0) / r)), int(round((y1 - y0) / r))
    # relleno simétrico hasta el mínimo del modelo
    pad_x = max(0, lado_min - ancho)
    pad_y = max(0, lado_min - alto)
    px0, py1 = x0 - (pad_x // 2) * r, y1 + (pad_y // 2) * r
    W, H = ancho + pad_x, alto + pad_y
    T = lado_tesela or config.LADO_TESELA_PX
    mitad = T * r // 2
    a_geo = Transformer.from_crs(epsg, 4326, always_xy=True)
    teselas = []
    for fila in range(0, H, T):
        for col in range(0, W, T):
            cx = px0 + col * r + mitad
            cy = py1 - fila * r - mitad
            lon, lat = a_geo.transform(cx, cy)
            teselas.append({'fila': fila, 'col': col, 'lat': lat, 'lon': lon,
                            'cx': cx, 'cy': cy})
    from rasterio.transform import from_origin
    return {'epsg': epsg, 'transform': from_origin(px0, py1, r, r), 'ancho': W,
            'alto': H, 'lado_tesela': T, 'teselas': teselas,
            # la ventana del analista dentro de la malla rellenada
            'ventana': (pad_y // 2, pad_x // 2, alto, ancho),
            'relleno': bool(pad_x or pad_y)}


def caja_latlon(transform, alto, ancho, epsg):
    """[[sur, oeste], [norte, este]] de una malla, como la pide Leaflet."""
    a_geo = Transformer.from_crs(epsg, 4326, always_xy=True)
    x0, y1 = transform.c, transform.f
    x1, y0 = x0 + ancho * transform.a, y1 + alto * transform.e
    lons, lats = a_geo.transform([x0, x1, x0, x1], [y0, y0, y1, y1])
    return [[min(lats), min(lons)], [max(lats), max(lons)]]


# ----------------------------------------------------------- Sentinel-2 real
def _vec(v):
    return np.atleast_1d(np.asarray(v))


def adquirir_tesela(t, desde, hasta, bandas, nubes_max, max_escenas):
    import cubo
    T = config.LADO_TESELA_PX
    c = cubo.create(lat=t['lat'], lon=t['lon'], collection=config.COLECCION_S2,
                    bands=list(bandas) + ['SCL'], start_date=desde, end_date=hasta,
                    edge_size=T, resolution=config.RESOLUCION_M,
                    query={'eo:cloud_cover': {'lt': float(nubes_max)}})
    if c.sizes.get('time', 0) == 0:
        return None
    nubes = _vec(c['eo:cloud_cover'].values).astype(float)
    orden = np.argsort(nubes)[:max_escenas]
    c = c.isel(time=np.sort(orden))
    a = np.asarray(c.values, dtype=np.float32)               # (t, b, y, x)
    if a.ndim == 3:
        a = a[None]
    nb = len(bandas)
    base = _vec(c['s2:processing_baseline'].values) if 's2:processing_baseline' in c.coords \
        else np.array(['00.00'] * a.shape[0])
    for i in range(a.shape[0]):
        try:
            b = float(str(base[i]))
        except ValueError:
            b = 0.0
        if b >= 4.0:
            a[i, :nb] -= config.DESPLAZAMIENTO_L2A
    scl = a[:, nb]
    ok = np.isfinite(scl) & ~np.isin(scl, config.SCL_MALAS)
    ok &= np.all(np.isfinite(a[:, :nb]) & (a[:, :nb] > 0), axis=1)
    bandas_t = np.where(ok[:, None], a[:, :nb], np.nan)
    with np.errstate(all='ignore'):
        comp = np.nanmedian(bandas_t, axis=0)                # (b, y, x)
    n_ok = ok.sum(axis=0)
    fechas = [str(x)[:10] for x in _vec(c.time.values)]
    plat = _vec(c['platform'].values) if 'platform' in c.coords else [''] * len(fechas)
    nub = _vec(c['eo:cloud_cover'].values).astype(float)
    return {'bandas': {b: comp[i] for i, b in enumerate(bandas)}, 'n_validas': n_ok,
            'x0': float(_vec(c.x.values)[0]), 'y0': float(_vec(c.y.values)[0]),
            'epsg': int(c.attrs.get('epsg', 0)),
            'escenas': [{'fecha': f, 'nubes_metadato': round(float(n), 1),
                         'plataforma': str(p)} for f, n, p in zip(fechas, nub, plat)]}


def adquirir_terreno(t, epsg, transform_tesela, lado):
    """Elevación (m) y pendiente (°) de una tesela, calculadas a 30 m con margen
    y llevadas a la malla de 10 m de la tesela."""
    import cubo
    from rasterio.warp import Resampling, reproject
    margen_m = 300
    lado_30 = int(math.ceil((lado * config.RESOLUCION_M + 2 * margen_m) / 30.0))
    d = cubo.create(lat=t['lat'], lon=t['lon'], collection=config.COLECCION_DEM,
                    bands=['data'], start_date=config.FECHAS_DEM[0],
                    end_date=config.FECHAS_DEM[1], edge_size=lado_30, resolution=30)
    if d.sizes.get('time', 0) == 0:
        raise DatosInvalidos('no hay modelo de elevación para esta zona')
    z = np.asarray(d.values, dtype=np.float32)
    while z.ndim > 2:
        z = np.nanmax(z, axis=0) if z.shape[0] > 1 else z[0]
    dy, dx = np.gradient(z, 30.0, 30.0)
    pend = np.degrees(np.arctan(np.hypot(dx, dy))).astype(np.float32)
    from rasterio.transform import from_origin
    src_tr = from_origin(float(_vec(d.x.values)[0]), float(_vec(d.y.values)[0]), 30, 30)
    if int(d.attrs.get('epsg', epsg)) != epsg:
        raise DatosInvalidos('el DEM y la imagen quedaron en zonas UTM distintas')
    salida = {}
    for nombre, arr in (('elevacion', z), ('pendiente', pend)):
        dst = np.full((lado, lado), np.nan, np.float32)
        reproject(arr, dst, src_transform=src_tr, src_crs=f'EPSG:{epsg}',
                  dst_transform=transform_tesela, dst_crs=f'EPSG:{epsg}',
                  resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
        salida[nombre] = dst
    return salida


# ---------------------------------------------------------- modo de prueba
def adquirir_tesela_falsa(t, desde, hasta, bandas, nubes_max, max_escenas):
    """Una tesela sintética en niveles digitales L2A: bosque con una cicatriz de
    suelo desnudo alargada ladera abajo. No sale a la red."""
    T = config.LADO_TESELA_PX
    rng = np.random.default_rng(abs(hash((round(t['cx']), round(t['cy'])))) % 2**32)
    yy, xx = np.mgrid[0:T, 0:T].astype(np.float32)
    gy, gx = yy + t['fila'], xx + t['col']
    cic = ((gx - 190) ** 2 / 14 ** 2 + (gy - 150) ** 2 / 55 ** 2) <= 1
    bosque = {'B01': 260, 'B02': 300, 'B03': 520, 'B04': 330, 'B08': 3100}
    suelo = {'B01': 900, 'B02': 1000, 'B03': 1250, 'B04': 1500, 'B08': 2200}
    comp = {}
    for b in bandas:
        base = np.full((T, T), bosque.get(b, 1000), np.float32)
        base[cic] = suelo.get(b, 1500)
        comp[b] = base * (1 + 0.06 * rng.standard_normal((T, T)).astype(np.float32))
    return {'bandas': comp, 'n_validas': np.full((T, T), 3, np.int16),
            'x0': t['cx'] - T * config.RESOLUCION_M / 2,
            'y0': t['cy'] + T * config.RESOLUCION_M / 2, 'epsg': 0,
            'escenas': [{'fecha': desde, 'nubes_metadato': 0.0,
                         'plataforma': 'sintetica'}]}


def terreno_falso(t, lado):
    yy, xx = np.mgrid[0:lado, 0:lado].astype(np.float32)
    z = 1200 + 0.35 * 10 * (yy + t['fila']) + 40 * np.sin((xx + t['col']) / 30.0)
    dy, dx = np.gradient(z, 10.0, 10.0)
    return {'elevacion': z, 'pendiente': np.degrees(np.arctan(np.hypot(dx, dy)))}


# ------------------------------------------------------------- el mosaico
def componer(plan, desde, hasta, bandas, con_terreno, nubes_max=40,
             max_escenas=None, avance=None):
    """Devuelve el mosaico de la malla del plan: bandas en niveles digitales,
    terreno si se pidió, máscara de válidos y el registro de escenas usadas."""
    from rasterio.transform import from_origin
    max_escenas = max_escenas or config.MAX_ESCENAS
    H, W, T = plan['alto'], plan['ancho'], plan['lado_tesela']
    r = config.RESOLUCION_M
    X0, Y0 = plan['transform'].c, plan['transform'].f
    mos = {b: np.full((H, W), np.nan, np.float32) for b in bandas}
    ter = {k: np.full((H, W), np.nan, np.float32) for k in ('elevacion', 'pendiente')} \
        if con_terreno else {}
    escenas, sin_escenas = {}, 0
    adquirir = adquirir_tesela_falsa if config.MODO_FALSO else adquirir_tesela
    n = len(plan['teselas'])
    for k, t in enumerate(plan['teselas']):
        if avance:
            avance(k / n, f'Tesela {k + 1} de {n}: buscando escenas Sentinel-2')
        d = adquirir(t, desde, hasta, bandas, nubes_max, max_escenas)
        if d is None:
            sin_escenas += 1
            continue
        # dónde cae de verdad, según las coordenadas que devolvió cubo
        c0 = int(round((d['x0'] - X0) / r))
        f0 = int(round((Y0 - d['y0']) / r))
        fs = slice(max(f0, 0), min(f0 + T, H))
        cs = slice(max(c0, 0), min(c0 + T, W))
        ls = slice(fs.start - f0, fs.stop - f0)
        lc = slice(cs.start - c0, cs.stop - c0)
        for b in bandas:
            bloque = d['bandas'][b][ls, lc]
            destino = mos[b][fs, cs]
            np.copyto(destino, bloque, where=np.isnan(destino) & np.isfinite(bloque))
        for e in d['escenas']:
            escenas.setdefault(e['fecha'], e)
        if con_terreno:
            if avance:
                avance((k + 0.6) / n, f'Tesela {k + 1} de {n}: relieve')
            tr_t = from_origin(X0 + c0 * r, Y0 - f0 * r, r, r)
            dt = terreno_falso(t, T) if config.MODO_FALSO else \
                adquirir_terreno(t, plan['epsg'], tr_t, T)
            for nombre in ter:
                bloque = dt[nombre][ls, lc]
                destino = ter[nombre][fs, cs]
                np.copyto(destino, bloque, where=np.isnan(destino) & np.isfinite(bloque))
    if sin_escenas == n:
        raise DatosInvalidos(
            f'no hay ninguna escena Sentinel-2 con menos de {nubes_max:.0f} % de nubes '
            f'entre {desde} y {hasta}. Amplía las fechas o sube la nubosidad máxima.')
    valido = np.all([np.isfinite(mos[b]) for b in bandas], axis=0)
    if con_terreno:
        valido &= np.isfinite(ter['pendiente']) & np.isfinite(ter['elevacion'])
    return {'bandas': mos, 'terreno': ter, 'valido': valido,
            'escenas': sorted(escenas.values(), key=lambda e: e['fecha']),
            'teselas': n, 'teselas_sin_escena': sin_escenas}


def recortar(arr, ventana):
    f, c, h, w = ventana
    return arr[..., f:f + h, c:c + w]
