"""
Cola de trabajos en memoria, con un solo trabajador.

La página crea un trabajo, recibe su id y consulta el avance cada pocos
segundos, como la barra de progreso de betaearth. Así ninguna petición HTTP dura
minutos, y una sola inferencia a la vez cabe en los 512 MB del plan gratuito.

Si Render reinicia o duerme el servicio, los trabajos en memoria se pierden: la
página lo detecta (404) y lo dice.
"""
import queue
import threading
import time
import traceback
import uuid

from . import config


class ColaLlena(RuntimeError):
    pass


class ErrorConCodigo(Exception):
    def __init__(self, codigo, detalle):
        super().__init__(detalle)
        self.codigo, self.detalle = codigo, detalle


_cola = queue.Queue()
_trabajos = {}
_lock = threading.Lock()
_hilo = None


def _limpiar():
    limite = time.time() - config.MINUTOS_RESULTADO * 60
    for k in [k for k, t in _trabajos.items()
              if t['estado'] in ('listo', 'error') and t['actualizado'] < limite]:
        _trabajos.pop(k, None)


def _trabajador(funcion):
    while True:
        tid = _cola.get()
        t = _trabajos.get(tid)
        if t is None:
            continue
        t.update(estado='procesando', inicio=time.time(), actualizado=time.time())

        def avance(frac, texto, t=t):
            t['progreso'] = round(max(t['progreso'], min(99.0, 100 * frac)), 1)
            t['paso'] = texto
            t['actualizado'] = time.time()

        try:
            resultado, archivos = funcion(t['peticion'], avance)
            t.update(estado='listo', progreso=100.0, paso='Listo', resultado=resultado,
                     archivos=archivos)
        except ErrorConCodigo as e:
            t.update(estado='error', codigo=e.codigo, error=e.detalle)
        except Exception as e:
            traceback.print_exc()
            t.update(estado='error', codigo=500, error=f'{type(e).__name__}: {e}')
        t['actualizado'] = time.time()
        t['segundos'] = round(time.time() - t['inicio'], 1)
        _cola.task_done()


def crear(peticion, funcion):
    global _hilo
    with _lock:
        _limpiar()
        pendientes = sum(1 for t in _trabajos.values()
                         if t['estado'] in ('en_cola', 'procesando'))
        if pendientes >= config.MAX_COLA:
            raise ColaLlena(f'hay {pendientes} trabajos en curso; prueba en un momento')
        if _hilo is None or not _hilo.is_alive():
            _hilo = threading.Thread(target=_trabajador, args=(funcion,), daemon=True)
            _hilo.start()
        tid = uuid.uuid4().hex[:12]
        _trabajos[tid] = {'id': tid, 'estado': 'en_cola', 'progreso': 0.0,
                          'paso': 'En cola', 'peticion': peticion, 'creado': time.time(),
                          'actualizado': time.time(), 'resultado': None,
                          'archivos': {}, 'error': None, 'codigo': None}
        _cola.put(tid)
        return tid, pendientes


def ver(tid):
    t = _trabajos.get(tid)
    if t is None:
        return None
    pos = None
    if t['estado'] == 'en_cola':
        pos = sum(1 for o in _trabajos.values()
                  if o['estado'] == 'en_cola' and o['creado'] < t['creado'])
    return {'id': t['id'], 'estado': t['estado'], 'progreso': t['progreso'],
            'paso': t['paso'], 'posicion': pos, 'error': t['error'],
            'codigo': t['codigo'], 'segundos': t.get('segundos'),
            'resultado': t['resultado']}


def archivo(tid, nombre):
    t = _trabajos.get(tid)
    if t is None or t['estado'] != 'listo':
        return None
    return t['archivos'].get(nombre)
