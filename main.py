"""
API de detección de deslizamientos · La Convención, Cusco

Se despliega en Render. El frontend es un HTML estático en un Space de Hugging
Face que llama a esta API por CORS: el Space es solo el lienzo, aquí ocurre
todo el cálculo.

  GET  /                  qué hay en este servidor
  GET  /salud             respuesta inmediata; la página la usa para despertarlo
  GET  /modelos           cada modelo, su receta, su forma y si está entrenado
  GET  /ambito            el límite de la provincia, para dibujarlo en el mapa
  POST /trabajos          encola una detección y devuelve su id (202)
  GET  /trabajos/{id}     avance y, al terminar, el resultado
  GET  /trabajos/{id}/zip el ZIP con GeoTIFF, GeoJSON, PNG y manifiesto
  POST /predict           lo mismo en una sola llamada, para scripts y pruebas
"""
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from nucleo import ambito, config, modelos, procesar, trabajos

app = FastAPI(title='Deslizamientos · La Convención', version='4.0')
app.add_middleware(CORSMiddleware, allow_origins=config.ORIGENES,
                   allow_credentials=False, allow_methods=['GET', 'POST', 'OPTIONS'],
                   allow_headers=['*'], max_age=3600)


class Peticion(BaseModel):
    sur: float = Field(..., ge=-90, le=90)
    oeste: float = Field(..., ge=-180, le=180)
    norte: float = Field(..., ge=-90, le=90)
    este: float = Field(..., ge=-180, le=180)
    desde: str = Field(..., description='AAAA-MM-DD')
    hasta: str = Field(..., description='AAAA-MM-DD')
    modelo: Optional[str] = None
    umbral: Optional[float] = Field(None, ge=0.05, le=0.95)
    pendiente_min: float = Field(0.0, ge=0.0, le=60.0)
    nubes_max: float = Field(40.0, ge=5.0, le=90.0)
    cobertura_min: Optional[float] = Field(None, ge=0.0, le=1.0)


def _http(e: trabajos.ErrorConCodigo):
    return HTTPException(e.codigo, e.detalle)


@app.get('/')
def raiz():
    return {'servicio': 'detección de deslizamientos, La Convención', 'version': app.version,
            'modelos': sorted(modelos.catalogo()), 'modo_falso': config.MODO_FALSO,
            'max_lado_km': config.MAX_LADO_KM, 'aviso_lado_km': config.AVISO_LADO_KM,
            'ambito': ambito.descripcion()}


@app.get('/salud')
def salud():
    return {'estado': 'ok', 'modelos_cargados': modelos.cargados(),
            'modo_falso': config.MODO_FALSO}


@app.get('/modelos')
def ver_modelos():
    return {n: modelos.ficha(n) for n in sorted(modelos.catalogo())}


@app.get('/ambito')
def ver_ambito():
    return ambito.descripcion()


@app.post('/trabajos', status_code=202)
def crear_trabajo(p: Peticion):
    try:
        pv = procesar.validar(p.model_dump())
    except trabajos.ErrorConCodigo as e:
        raise _http(e)
    try:
        tid, antes = trabajos.crear(pv, procesar.procesar)
    except trabajos.ColaLlena as e:
        raise HTTPException(429, str(e))
    return {'id': tid, 'trabajos_antes': antes}


@app.get('/trabajos/{tid}')
def ver_trabajo(tid: str):
    t = trabajos.ver(tid)
    if t is None:
        raise HTTPException(404, 'ese trabajo no existe o ya expiró; si el servidor se '
                                 'reinició, vuelve a lanzarlo')
    return t


@app.get('/trabajos/{tid}/zip')
def zip_trabajo(tid: str):
    a = trabajos.archivo(tid, 'zip')
    if a is None:
        raise HTTPException(404, 'no hay ZIP para ese trabajo')
    return Response(a[0], media_type=a[1], headers={
        'Content-Disposition': f'attachment; filename="deslizamientos_{tid}.zip"'})


@app.post('/predict')
def predecir(p: Peticion):
    try:
        resultado, archivos = procesar.procesar(p.model_dump())
    except trabajos.ErrorConCodigo as e:
        raise _http(e)
    return resultado


@app.exception_handler(HTTPException)
def error_http(request, exc):
    return JSONResponse(status_code=exc.status_code,
                        content={'ok': False, 'codigo': exc.status_code,
                                 'detalle': exc.detail})
