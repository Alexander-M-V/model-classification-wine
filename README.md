# Clasificación de Variedades de Vino 🍷📊

## Descripción del Proyecto
Este proyecto implementa un sistema de clasificación de variedades de vino utilizando técnicas de aprendizaje automático. Se emplean múltiples algoritmos para predecir la variedad de vino basándose en características químicas.

## Características Principales
- Procesamiento de datos del conjunto de datos (CSV): https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data 
- Preprocesamiento de datos (limpieza, escalado)
- Entrenamiento de modelos:
  - Random Forest
  - Gradient Boosting
  - Support Vector Machine (SVM)
- Seguimiento de experimentos con MLflow
- Visualización comparativa de rendimiento de modelos

## Requisitos
- Python 3.8+
- Librerías:
  - pandas
  - numpy
  - scikit-learn
  - mlflow
  - matplotlib
  - seaborn

## Instalación
```bash
git clone https://github.com/tu-usuario/model-wine.git
cd model-wine
pip install -r requirements.txt
```

## Uso
```bash
python model-wine.py
```

## Métricas de Evaluación
- Precisión
- Recall
- F1-Score
- Accuracy

## Resultados
El script genera:
- Métricas de rendimiento de modelos
- Gráfico comparativo de modelos
- Predicciones para nuevas muestras

## Contribuciones
¡Las contribuciones son bienvenidas! Por favor, abre un issue o envía un pull request.

## Licencia
MIT License
