"""
Salidas: GeoTIFF en UTM para análisis, GeoJSON y capas PNG en grados para el mapa.

Las capas PNG se reproyectan a EPSG:4326 antes de pintarlas. Una malla UTM
puesta tal cual sobre Leaflet queda girada por la convergencia de meridianos, y
cerca del borde de una zona, como Quillabamba en la 18, eso son decenas de
metros en el extremo de un área de pocos kilómetros.
"""
import io
import json
import zipfile

import numpy as np
from PIL import Image
from pyproj import Transformer


def geotiff(arr, transform, epsg, dtype, nodata=None, descripcion=None):
    import rasterio
    a = np.asarray(arr)
    if a.ndim == 2:
        a = a[None]
    perfil = dict(driver='GTiff', height=a.shape[1], width=a.shape[2], count=a.shape[0],
                  dtype=dtype, crs=f'EPSG:{epsg}', transform=transform,
                  compress='deflate', tiled=True, blockxsize=256, blockysize=256)
    if nodata is not None:
        perfil['nodata'] = nodata
    if a.shape[1] < 256 or a.shape[2] < 256:
        perfil.pop('tiled'); perfil.pop('blockxsize'); perfil.pop('blockysize')
    with rasterio.MemoryFile() as mf:
        with mf.open(**perfil) as ds:
            ds.write(a.astype(dtype))
            if descripcion:
                ds.set_band_description(1, descripcion)
        return mf.read()


def a_grados(arr, transform, epsg, nearest=True):
    """Reproyecta una capa de la malla UTM a una malla regular en grados.
    Devuelve (capa, [[sur, oeste], [norte, este]])."""
    from rasterio.warp import Resampling, calculate_default_transform, reproject
    h, w = arr.shape
    x0, y1 = transform.c, transform.f
    x1, y0 = x0 + w * transform.a, y1 + h * transform.e
    dst_tr, dw, dh = calculate_default_transform(
        f'EPSG:{epsg}', 'EPSG:4326', w, h, left=x0, bottom=y0, right=x1, top=y1)
    dst = np.full((dh, dw), np.nan, np.float32)
    reproject(arr.astype(np.float32), dst, src_transform=transform,
              src_crs=f'EPSG:{epsg}', dst_transform=dst_tr, dst_crs='EPSG:4326',
              resampling=Resampling.nearest if nearest else Resampling.bilinear,
              src_nodata=np.nan, dst_nodata=np.nan)
    oeste, norte = dst_tr.c, dst_tr.f
    este, sur = oeste + dw * dst_tr.a, norte + dh * dst_tr.e
    return dst, [[sur, oeste], [norte, este]]


def _png(rgba):
    b = io.BytesIO()
    Image.fromarray(rgba, 'RGBA').save(b, 'PNG', optimize=True)
    return b.getvalue()


def png_mascara(m_grados):
    h, w = m_grados.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    si = np.nan_to_num(m_grados, nan=0) > 0.5
    rgba[si] = (230, 40, 60, 255)
    return _png(rgba)


# paleta de probabilidad: transparente bajo 0.15, de amarillo a rojo oscuro
_PUNTOS = [(0.15, (255, 237, 160, 0)), (0.3, (254, 217, 118, 150)),
           (0.5, (253, 141, 60, 200)), (0.7, (240, 59, 32, 230)),
           (1.0, (150, 0, 38, 255))]


def png_probabilidad(p_grados):
    p = np.nan_to_num(p_grados, nan=0.0)
    h, w = p.shape
    rgba = np.zeros((h, w, 4), np.float32)
    xs = [q for q, _ in _PUNTOS]
    for k in range(4):
        ys = [c[k] for _, c in _PUNTOS]
        rgba[..., k] = np.interp(p, xs, ys)
    rgba[p < _PUNTOS[0][0]] = 0
    return _png(rgba.astype(np.uint8))


def png_color_natural(b04, b03, b02, valido=None):
    """Color natural de la mediana sin nubes, estirado de 0 a 1600 niveles
    digitales con corrección gamma: es lo que el modelo vio."""
    rgb = np.stack([b04, b03, b02], axis=-1).astype(np.float32)
    rgb = np.clip(np.nan_to_num(rgb, nan=0.0) / 1600.0, 0, 1) ** (1 / 1.6)
    h, w = b04.shape
    a = np.full((h, w), 255, np.uint8)
    if valido is not None:
        a[~valido] = 0
    a[~np.isfinite(b04)] = 0
    return _png(np.dstack([(rgb * 255).astype(np.uint8), a]))


def geojson_de_mascara(m, transform, epsg, min_px=4):
    """Polígonos de la máscara, en grados, con su área en hectáreas."""
    from rasterio.features import shapes
    from shapely.geometry import mapping, shape
    from shapely.ops import transform as sh_tr
    a_geo = Transformer.from_crs(epsg, 4326, always_xy=True)
    feats = []
    px_ha = abs(transform.a * transform.e) / 1e4
    for geom, val in shapes(m.astype(np.uint8), mask=m.astype(bool), transform=transform):
        if val != 1:
            continue
        g = shape(geom)
        n_px = g.area / abs(transform.a * transform.e)
        if n_px < min_px:
            continue
        g4326 = sh_tr(lambda x, y, z=None: a_geo.transform(x, y), g)
        feats.append({'type': 'Feature',
                      'properties': {'area_ha': round(n_px * px_ha, 3), 'px': int(round(n_px))},
                      'geometry': mapping(g4326)})
    feats.sort(key=lambda f: -f['properties']['area_ha'])
    for i, f in enumerate(feats, 1):
        f['properties']['id'] = i
    return {'type': 'FeatureCollection', 'features': feats}


def armar_zip(piezas):
    b = io.BytesIO()
    with zipfile.ZipFile(b, 'w', zipfile.ZIP_DEFLATED) as z:
        for nombre, contenido in piezas:
            if isinstance(contenido, (dict, list)):
                contenido = json.dumps(contenido, ensure_ascii=False, indent=1).encode()
            z.writestr(nombre, contenido)
    return b.getvalue()
