# Modelos

Cada modelo son dos archivos con el mismo nombre de base:

```
deeplabv3plus.onnx    el grafo y los pesos
deeplabv3plus.json    la receta: canales, normalización, salida, umbral
```

Los dos salen juntos de `herramientas/exportar_onnx.py`, que carga el
checkpoint de la Fase 4 en modo estricto, comprueba la paridad con PyTorch y
se niega a exportar si los pesos quedaron en la inicialización.

Al arrancar, el servidor revisa los pesos de cada ONNX. Uno exportado sin
cargar el entrenamiento aparece en `/modelos` como `sin_entrenar` y no se usa.
Así estaban los dos ONNX del repositorio original.

El DeepLabV3+ exportado pesa unos 90 MB y la U-Net con ResNet-34 unos 98 MB:
los dos caben bajo el límite de 100 MB de GitHub, así que se suben como
archivos normales, sin Git LFS. Si alguna vez llegan como punteros de LFS, el
servidor los detecta y los baja de `URL_PESOS_BASE`.
