"""
Todo lo configurable del backend, en un solo sitio.

Cada valor se puede cambiar con una variable de entorno del mismo nombre, que es
como se ajusta en Render (Environment) sin tocar el código.
"""
import json
import os


def _env(nombre, defecto):
    return os.environ.get(nombre, defecto)


RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------------- modelos
# Cada modelo es un .onnx más una receta .json con el mismo nombre de base, que
# dice qué canales recibe, cómo se normalizan, qué salida tiene y su umbral.
CARPETA_MODELOS = _env('CARPETA_MODELOS', os.path.join(RAIZ, 'modelos'))
# Si un .onnx llega como puntero de Git LFS, se baja de aquí: <URL>/<archivo>
URL_PESOS_BASE = _env('URL_PESOS_BASE', '')
# Un ONNX con pesos de inicialización no detecta nada aunque corra. Se rechaza
# salvo que esto valga 1, que solo sirve para probar el circuito.
PERMITIR_SIN_ENTRENAR = _env('PERMITIR_SIN_ENTRENAR', '0') == '1'
HILOS_ONNX = int(_env('HILOS_ONNX', '1'))
# En el plan gratuito de Render hay 512 MB: mejor un modelo cargado a la vez.
UN_MODELO_A_LA_VEZ = _env('UN_MODELO_A_LA_VEZ', '1') == '1'

# ----------------------------------------------------------------------- CORS
# La URL del Space, p. ej. https://usuario-espacio.hf.space. '*' para probar.
ORIGENES = [o.strip() for o in _env('ORIGENES_PERMITIDOS', '*').split(',') if o.strip()]

# ---------------------------------------------------------- fuente de imágenes
STAC = 'https://planetarycomputer.microsoft.com/api/stac/v1'
COLECCION_S2 = 'sentinel-2-l2a'
COLECCION_DEM = 'cop-dem-glo-30'
# El DEM de Copernicus no es una serie temporal: todas sus teselas en Planetary
# Computer llevan la fecha 2021-04-22. Pedirlo con las fechas del analista no
# devuelve nada, así que va con una ventana propia.
FECHAS_DEM = (_env('DEM_DESDE', '2020-01-01'), _env('DEM_HASTA', '2023-12-31'))
RESOLUCION_M = 10
# Lado de cada cubo de cubo. 256 px son 2.56 km: pocas llamadas y poca memoria.
LADO_TESELA_PX = int(_env('LADO_TESELA_PX', '256'))
# Escenas por tesela, las menos nubosas según el metadato. Acota memoria y tiempo.
MAX_ESCENAS = int(_env('MAX_ESCENAS', '6'))
# Clases SCL que se descartan: sin dato, saturado, sombra de nube, nube media,
# nube alta y cirro.
SCL_MALAS = (0, 1, 3, 8, 9, 10)
# Desde la baseline 04.00 (enero de 2022) los niveles digitales de L2A llevan
# sumado 1000. Se resta por escena.
DESPLAZAMIENTO_L2A = 1000.0

# ---------------------------------------------- unidades de Landslide4Sense
# Landslide4Sense trae cada banda dividida por su media de escena. Estas son las
# medias de La Convención (reflectancia L2A en niveles digitales), medidas con
# herramientas/calibrar_escala.py sobre la provincia en temporada seca.
_RUTA_ESCALA = _env('RUTA_ESCALA', os.path.join(RAIZ, 'escala_laconvencion.json'))


def _cargar_escala():
    try:
        with open(_RUTA_ESCALA, encoding='utf-8') as f:
            return json.load(f)['media_dn']
    except Exception:
        return None


ESCALA_L4S = _cargar_escala()

# ------------------------------------------------------------------- límites
MAX_LADO_KM = float(_env('MAX_LADO_KM', '12'))
AVISO_LADO_KM = float(_env('AVISO_LADO_KM', '6'))
SOLAPE = float(_env('SOLAPE', '0.5'))
COBERTURA_MIN = float(_env('COBERTURA_MIN', '0.30'))
MAX_COLA = int(_env('MAX_COLA', '4'))
MINUTOS_RESULTADO = int(_env('MINUTOS_RESULTADO', '30'))

# -------------------------------------------------------------------- ámbito
#   'bloquear': rechaza con 403 lo que caiga fuera. Es lo que va en la entrega.
#   'avisar'  : deja correr y marca el resultado como fuera del ámbito.
#   'libre'   : sin límite, solo para pruebas.
AMBITO = _env('AMBITO', 'bloquear')
NOMBRE_AMBITO = 'provincia de La Convención, Cusco'
RUTA_LIMITE = _env('RUTA_LIMITE', os.path.join(RAIZ, 'limite_laconvencion.geojson'))
BBOX_ESTUDIO = (-13.4489, -73.9805, -11.1988, -71.9730)   # sur, oeste, norte, este
MIN_DENTRO = float(_env('MIN_DENTRO', '0.5'))

# --------------------------------------------------------------------- pruebas
# A 1 no sale a la red: fabrica un área sintética con una cicatriz.
MODO_FALSO = _env('MODO_FALSO', '0') == '1'
CENTRO_POR_DEFECTO = (-12.86, -72.69)    # Quillabamba
