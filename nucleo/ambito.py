"""
El ámbito de estudio: la provincia de La Convención, Cusco.

Sin esto, el artefacto corre en cualquier parte del mundo y devuelve un mapa con
la misma apariencia que uno válido. Los modelos no se entrenaron ni se validaron
fuera de la provincia, y el alcance de la tesis lo dice explícitamente, así que
el artefacto debe hacer cumplir su propio límite en vez de confiar en que nadie
dibuje fuera.

Usa el polígono exacto si hay un GeoJSON disponible, y si no, la caja de la
provincia. La caja es más permisiva: deja pasar esquinas que en realidad caen en
Calca o en La Mar. Está declarado para que nadie lo descubra después.
"""
import json
import os

from . import config

_LIMITE = None
_CARGADO = False


def _cargar():
    """El polígono de la provincia, si está. Se lee una sola vez."""
    global _LIMITE, _CARGADO
    if _CARGADO:
        return _LIMITE
    _CARGADO = True
    ruta = config.RUTA_LIMITE
    if not ruta or not os.path.exists(ruta):
        print(f'[ambito] sin {os.path.basename(ruta or "limite")}, se usa la caja')
        return None
    try:
        from shapely.geometry import shape
        from shapely.ops import unary_union
        with open(ruta, encoding='utf-8') as f:
            gj = json.load(f)
        if gj.get('type') == 'FeatureCollection':
            geos = [shape(f['geometry']) for f in gj['features'] if f.get('geometry')]
        elif gj.get('type') == 'Feature':
            geos = [shape(gj['geometry'])]
        else:
            geos = [shape(gj)]
        _LIMITE = unary_union(geos)
        print(f'[ambito] polígono cargado de {os.path.basename(ruta)}')
    except Exception as e:
        print(f'[ambito] no se pudo leer el límite ({type(e).__name__}), '
              'se usa la caja')
        _LIMITE = None
    return _LIMITE


def fraccion_dentro(bbox):
    """Qué parte del rectángulo cae dentro del ámbito, entre 0 y 1."""
    sur, oeste, norte, este = bbox
    from shapely.geometry import box as _box
    rect = _box(oeste, sur, este, norte)
    if rect.area <= 0:
        return 0.0
    lim = _cargar()
    if lim is None:
        s, o, n, e = config.BBOX_ESTUDIO
        lim = _box(o, s, e, n)
    return float(rect.intersection(lim).area/rect.area)


def revisar(bbox):
    """(permitido, fuera_del_ambito, fraccion, mensaje)."""
    frac = fraccion_dentro(bbox)
    dentro = frac >= config.MIN_DENTRO
    if dentro:
        return True, False, frac, None
    msg = (f'El área dibujada queda fuera del ámbito del estudio: solo el '
           f'{frac*100:.0f} % cae dentro de la {config.NOMBRE_AMBITO}, y se '
           f'exige al menos el {config.MIN_DENTRO*100:.0f} %. Los modelos no se '
           'entrenaron ni se validaron fuera de esa provincia, así que un '
           'resultado ahí no tendría respaldo.')
    if config.AMBITO == 'bloquear':
        return False, True, frac, msg
    if config.AMBITO == 'avisar':
        return True, True, frac, msg
    return True, False, frac, None


def descripcion():
    """Lo que el frontend necesita para dibujar el límite en el mapa."""
    lim = _cargar()
    s, o, n, e = config.BBOX_ESTUDIO
    d = {'modo': config.AMBITO, 'nombre': config.NOMBRE_AMBITO,
         'min_dentro': config.MIN_DENTRO,
         'bbox': {'sur': s, 'oeste': o, 'norte': n, 'este': e},
         'tipo': 'poligono' if lim is not None else 'caja'}
    if lim is not None:
        from shapely.geometry import mapping
        d['geojson'] = {'type': 'Feature', 'properties': {},
                        'geometry': mapping(lim)}
    return d
