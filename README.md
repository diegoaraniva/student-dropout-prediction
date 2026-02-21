# Dashboard de Prediccion de Desercion Estudiantil

Dashboard interactivo desarrollado con Streamlit para predecir la desercion estudiantil mediante modelos de Machine Learning.

## Caracteristicas

- Comparacion de 3 modelos de clasificacion:
  - Regresion Logistica
  - K-Nearest Neighbors
  - XGBoost
  
- Soporte para dos datasets:
  - Dataset Principal (2015-2019)
  - Dataset Completo

- Visualizaciones interactivas:
  - Metricas de rendimiento
  - Matrices de confusion
  - Curvas ROC y Precision-Recall
  - Feature importance (XGBoost)

## Requisitos

- Python 3.8 o superior
- Las librerias especificadas en `requirements.txt`

## Instalacion

1. Asegurate de tener los archivos de datos en el directorio raiz del proyecto:
   - `Tbl_DesercionEstudiantil_PrimerAnio_2015_2019.csv`
   - `Tbl_DesercionEstudiantil_PrimerAnio_.csv`

2. Instala las dependencias:

```bash
pip install -r requirements.txt
```

## Ejecucion Local

Desde el directorio raiz del proyecto (no desde build/), ejecuta:

```bash
streamlit run build/app.py
```

El dashboard se abrira automaticamente en tu navegador en `http://localhost:8501`

## Uso

1. Selecciona el dataset en la barra lateral
2. Ajusta los parametros:
   - Numero maximo de muestras
   - Tamano del conjunto de prueba
3. Presiona "Entrenar Modelos"
4. Explora los resultados y analisis detallados

## Estructura del Proyecto

```
build/
├── app.py                  # Aplicacion principal de Streamlit
├── requirements.txt        # Dependencias de Python
└── README.md              # Este archivo
```

## Notas Importantes

- El entrenamiento puede tardar varios minutos dependiendo del tamano del dataset
- Se recomienda usar el dataset principal (2015-2019) para pruebas rapidas
- Los modelos se entrenan cada vez que presionas el boton (no hay cache de modelos)
- Asegurate de tener suficiente memoria RAM para datasets grandes

