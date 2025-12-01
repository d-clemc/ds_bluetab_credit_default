# 📘 Bluetab Test | Pipeline PAY_AMT4 (Kedro)
*Modelado predictivo con procesamiento, entrenamiento, evaluación y reporting automatizado*

Este repositorio contiene la implementación completa del pipeline **PAY_AMT4**, desarrollado con *Kedro*, cuyo objetivo es construir un flujo reproducible de preprocesamiento → modelado → evaluación → visualizaciones utilizando modelos de regresión para predecir el pago realizado en el mes 4 (PAY_AMT4) en el dataset **Default of Credit Card Clients**.

También, se realiza modelo de predicción para **default.payment.next.month**, ubicado en:

notebooks/eda_default_payment_next_month.ipynb

## 🚀 Estructura del Pipeline

El proyecto implementa 4 pipelines modulares en Kedro:

```
payamt4_preprocessing → payamt4_modeling → payamt4_evaluating → payamt4_reporting
```

### 1. **Preprocesamiento** (`payamt4_preprocessing`)
- **Objetivo**: Limpieza y feature engineering sin data leakage
- **Input**: Dataset crudo Excel (`default of credit card clients.xls`)
- **Output**: Datos preprocesados (`payamt4_preprocessed_data.csv`)
- **Funciones**:
  - Winsorizing (1%-99%) en montos para reducir outliers
  - Feature engineering: ratios de pago, tendencias, estadísticas de retrasos
  - Limpieza categórica de EDUCATION y MARRIAGE
  - Prevención de data leakage (solo datos hasta mayo para predecir PAY_AMT4)

### 2. **Modelado** (`payamt4_modeling`)
- **Objetivo**: Entrenamiento de modelos de regresión con validación cruzada
- **Input**: Datos preprocesados + parámetros (test_size, random_state)
- **Output**: Modelo entrenado, resultados CV, métricas, datasets train/test
- **Funciones**:
  - Split train/test estratificado
  - Cross-validation con múltiples modelos (LinearRegression, RandomForest, XGBoost)
  - Selección del mejor modelo por R² promedio
  - Guardado de modelo y conjuntos de entrenamiento/prueba

### 3. **Evaluación** (`payamt4_evaluating`)
- **Objetivo**: Evaluación comprehensiva del modelo en datos de prueba
- **Input**: Datos preprocesados + modelo entrenado
- **Output**: Métricas de evaluación + predicciones
- **Funciones**:
  - Cálculo de métricas: MSE, RMSE, MAE, R², MAPE
  - Generación de predicciones sobre conjunto de prueba
  - Análisis de residuos y distribuciones

### 4. **Reporting** (`payamt4_reporting`)
- **Objetivo**: Generación automática de visualizaciones y reportes
- **Input**: Datos, modelo, predicciones + directorio de salida
- **Output**: Reportes visuales en PNG
- **Funciones**:
  - Gráfico de predicciones vs valores reales
  - Análisis de residuos (scatter + histograma)
  - Feature importance del modelo
  - Learning curves
  - Comparación de distribuciones

## 📊 Arquitectura de Datos

### Estructura de Directorios
```
data/
├── 01_raw/                     # Datos originales sin procesar
│   └── default of credit card clients.xls
├── 02_intermediate/            # Datos preprocessados
│   └── payamt4_preprocessed_data.csv
├── 05_model_input/            # Datasets train/test split
│   ├── X_train_payamt4.csv
│   ├── X_test_payamt4.csv
│   ├── y_train_payamt4.csv
│   └── y_test_payamt4.csv
├── 06_models/                 # Modelos entrenados
│   └── payamt4_model.pkl
├── 07_model_output/          # Resultados y métricas
│   ├── payamt4_cv_results.json
│   ├── payamt4_eval_metrics.json
│   ├── payamt4_metrics.json
│   └── payamt4_predictions.csv
└── 08_reporting/             # Reportes y visualizaciones
    └── payamt4_reports/
        ├── distribution_comparison.png
        ├── feature_importance.png
        ├── learning_curve.png
        ├── pred_vs_real.png
        ├── residual_hist.png
        └── residual_plot.png
```

## 🛠️ Comandos de Ejecución

### Ejecutar Pipeline Completo
```bash
# Ejecutar todos los pipelines en secuencia
kedro run
```

### Ejecutar Pipelines Individuales
```bash
# Solo preprocesamiento
kedro run --pipeline payamt4_preprocessing

# Solo modelado (requiere datos preprocesados)
kedro run --pipeline payamt4_modeling

# Solo evaluación (requiere modelo entrenado)
kedro run --pipeline payamt4_evaluating

# Solo reporting (requiere evaluación completa)
kedro run --pipeline payamt4_reporting
```

### Ejecutar Desde un Nodo Específico
```bash
# Ejecutar desde modelado en adelante
kedro run --from-nodes train_payamt4_model_node

# Ejecutar hasta evaluación
kedro run --to-nodes evaluate_payamt4_model_node

# Ejecutar nodos específicos
kedro run --nodes preprocess_payamt4_node,train_payamt4_model_node
```

### Visualización y Debugging
```bash
# Visualizar el pipeline como gráfico
kedro viz

# Ver estructura del pipeline
kedro pipeline list

# Describir un pipeline específico
kedro pipeline describe payamt4_modeling

# Ver catálogo de datos
kedro catalog list

# Información detallada de un dataset
kedro catalog describe payamt4_preprocessed_data
```

### Ejecución Paralela y Configuración
```bash
# Ejecución con configuración específica
kedro run --env local

# Forzar re-ejecución (ignorar cache)
kedro run --pipeline payamt4_modeling --force

# Ejecutar con parámetros personalizados
kedro run --params test_size:0.3,random_state:123
```

## 🔧 Configuración

### Parámetros Principales (`conf/base/parameters.yml`)
```yaml
test_size: 0.2              # Proporción del conjunto de prueba
random_state: 42            # Semilla para reproducibilidad
report_output_dir: data/08_reporting/payamt4_reports  # Directorio de reportes
```

### Credenciales (`conf/local/credentials.yml`)
Archivo para configuraciones sensibles (no versionado):
```yaml
# Agregar credenciales de APIs, bases de datos, etc.
```

## 📈 Resultados y Métricas

El pipeline genera automáticamente:

1. **Métricas de Cross-Validation** (`payamt4_cv_results.json`)
   - Comparación de modelos: Linear Regression, Random Forest, XGBoost
   - R² promedio por modelo y fold

2. **Métricas de Evaluación** (`payamt4_eval_metrics.json`)
   - MSE, RMSE, MAE, R², MAPE en conjunto de prueba

3. **Visualizaciones Automáticas**
   - Predicciones vs valores reales
   - Análisis de residuos
   - Importancia de características
   - Curvas de aprendizaje
   - Comparación de distribuciones

## 🔄 Flujo de Desarrollo

1. **Desarrollo**: Modificar nodos en `src/bluetab_test/pipelines/`
2. **Testing**: `kedro run --pipeline <pipeline_name>`
3. **Validación**: Revisar outputs en `data/07_model_output/`
4. **Visualización**: Generar reportes con `kedro run --pipeline payamt4_reporting`
5. **Iteración**: Ajustar parámetros y repetir

## 📋 Prerequisitos

```bash
# Instalar dependencias
pip install -r requirements.txt

# O usar el proyecto como paquete
pip install -e .
```

**Dependencias principales:**
- Kedro ~1.1.1
- pandas ≥2.3.3
- scikit-learn ≥1.7.2
- xgboost ≥3.1.2
- matplotlib, seaborn (visualización)