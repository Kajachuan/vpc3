# VPC3: Vision Transformer Classification Pipeline

Un proyecto integral para entrenar y evaluar modelos Vision Transformer (ViT) y otros modelos de clasificación de imágenes en el dataset Galaxy10.

## 🎯 Descripción General

VPC3 proporciona un pipeline completo para:
- **Entrenar** modelos de visión desde Hugging Face (DeiT, ViT, ConvNeXt, MobileViT, Swin)
- **Evaluar** con métricas detalladas (precisión, recall, F1, matriz de confusión)
- **Monitorear** experimentos con MLflow
- **Generar visualizaciones** de mapas de atención (Attention maps)
- **Exportar y servir** modelos con Streamlit

## 📊 Características

- ✅ Soporte para múltiples arquitecturas de Hugging Face
- ✅ Transformaciones de datos avanzadas (rotación, contraste, morfología)
- ✅ Seguimiento de experimentos con MLflow
- ✅ Early stopping y validación cruzada
- ✅ Métricas completas de evaluación
- ✅ Notebooks interactivos para exploración
- ✅ Interfaz Streamlit para inferencia

## 🚀 Inicio Rápido

### Requisitos Previos

- Python 3.8+
- pip o conda

### Instalación

1. Clona el repositorio:
```bash
git clone <repository-url>
cd vpc3
```

2. Crea un entorno virtual (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instala las dependencias:
```bash
pip install -r requirements.txt
```

### Uso Básico

Entrena un modelo con una configuración predefinida:

```bash
python app/main.py --config configs/deit-small/config.json
```

O crea tu propia configuración personalizada en `configs/tu-modelo/config.json`.

## 📁 Estructura del Proyecto

```
vpc3/
├── app/                          # Aplicación principal
│   ├── __init__.py
│   └── main.py                  # Punto de entrada del entrenamiento
│
├── src/vit/                      # Código fuente principal
│   ├── data/                     # Carga y gestión de datos
│   │   ├── __init__.py
│   │   └── data_loader.py       # DataLoader para Galaxy10
│   ├── models/                   # Modelos
│   │   ├── __init__.py
│   │   └── models.py            # Carga modelos desde HF
│   ├── train/                    # Entrenamiento
│   │   ├── __init__.py
│   │   └── trainer.py           # Trainer con MLflow
│   ├── eval/                     # Evaluación
│   ├── inference/                # Inferencia
│   ├── metrics/                  # Métricas
│   │   ├── __init__.py
│   │   └── metrics.py           # Cálculo de métricas
│   ├── transforms/               # Transformaciones de datos
│   │   ├── __init__.py
│   │   └── transforms.py        # Augmentación de imágenes
│   └── utils/                    # Utilidades
│       └── utils.py
│
├── configs/                      # Configuraciones por modelo
│   ├── deit-small/config.json
│   ├── swin-tiny/config.json
│   ├── convnext-tiny/config.json
│   ├── mobilevit-small/config.json
│   └── ...
│
├── notebooks/                    # Análisis interactivos
│   ├── 1. Data visualization.ipynb
│   ├── 2. Train.ipynb
│   └── 3. Attention map.ipynb
│
├── data/                         # Datos
│   ├── raw/                     # Datos originales
│   ├── interim/                 # Datos procesados temporalmente
│   └── processed/               # Datos finales
│
├── checkpoints/                  # Modelos entrenados
├── logs/                         # Logs de entrenamiento
├── experiments/                  # Resultados de experimentos
├── scripts/                      # Scripts utilitarios
├── tests/                        # Tests unitarios e integración
├── examples/                     # Ejemplos y plantillas
├── docs/                         # Documentación
├── requirements.txt              # Dependencias
└── README.md
```

## 📋 Modelos Soportados

El proyecto soporta cualquier modelo de clasificación de imágenes de Hugging Face. Configuraciones predefinidas incluyen:

| Modelo | Checkpoint | Configuración |
|--------|-----------|---------------|
| DeiT Small | `facebook/deit-small-patch16-224` | ✅ |
| Swin Tiny | `microsoft/swin-tiny-patch4-window7-224` | ✅ |
| ConvNeXt Tiny | `facebook/convnext-tiny-224` | ✅ |
| MobileViT Small | `apple/mobilevit-small` | ✅ |

## ⚙️ Configuración

Cada modelo tiene su propia configuración JSON. Ejemplo (`configs/deit-small/config.json`):

```json
{
  "checkpoint": "facebook/deit-small-patch16-224",
  "batch_size": 16,
  "epochs": 20,
  "learning_rate": 5e-5,
  "early_stopping_patience": 3,
  "img_height": 224,
  "img_width": 224,
  "morph_kernel_size": [7, 7],
  "rotation_degrees": 180,
  "contrast": 0.2,
  "translate": [0.1, 0.1]
}
```

### Parámetros Configurables

- **checkpoint**: Identificador del modelo en Hugging Face
- **batch_size**: Tamaño del lote para entrenamiento
- **epochs**: Número de épocas
- **learning_rate**: Tasa de aprendizaje
- **early_stopping_patience**: Paciencia para early stopping
- **img_height/img_width**: Dimensiones de entrada de imagen
- **morph_kernel_size**: Tamaño del kernel para operaciones morfológicas
- **rotation_degrees**: Ángulos de rotación en augmentación
- **contrast**: Factor de contraste en augmentación
- **translate**: Rango de traslación en pixeles

## 📚 Dataset

El proyecto utiliza el dataset **Galaxy10**, que contiene 17,736 imágenes de galaxias clasificadas en 10 categorías:

0. Disturbed
1. Merging
2. Round Smooth
3. Smooth, Cigar shaped
4. Cigar Shaped Smooth
5. Barred Spiral
6. Unbarred Tight Spiral
7. Unbarred Loose Spiral
8. Edge-on without Bulge
9. Edge-on with Bulge

Se descarga automáticamente usando el dataset de Hugging Face.

## 🔧 Dependencias Principales

- **torch==2.8.0**: Framework de aprendizaje profundo
- **torchvision==0.23.0**: Modelos y utilidades de visión por computadora
- **transformers==4.37.2**: Modelos preentrenados de Hugging Face
- **datasets==4.0.0**: Carga de datasets
- **scikit-learn==1.6.1**: Métricas y utilidades ML
- **matplotlib==3.10.0**: Visualización
- **pandas==2.2.2**: Manejo de datos tabulares
- **opencv-python==4.12.0.88**: Procesamiento de imágenes
- **accelerate==0.28.0**: Aceleración distribuida

Ver `requirements.txt` para la lista completa.

## 🎓 Notebooks

El proyecto incluye notebooks interactivos para exploración:

1. **Data visualization.ipynb**: Análisis exploratorio del dataset
2. **Train.ipynb**: Proceso de entrenamiento paso a paso
3. **Attention map.ipynb**: Visualización de mapas de atención

Ejecuta con Jupyter:
```bash
jupyter notebook
```

## 📊 Monitoreo con MLflow

Los experimentos se registran automáticamente en MLflow. Para ver el dashboard:

```bash
mlflow ui
```

Luego abre `http://localhost:5000` en tu navegador.

## 📈 Evaluación y Métricas

El proyecto calcula automáticamente:

- **Accuracy**: Precisión global
- **Precision, Recall, F1-Score**: Por clase y global ponderadas
- **Confusion Matrix**: Matriz de confusión

## 📝 Flujo de Trabajo Típico

1. **Configuración**: Crea o edita `configs/mi-modelo/config.json`
2. **Entrenamiento**: `python app/main.py --config configs/mi-modelo/config.json`
3. **Monitoreo**: Abre MLflow para ver métricas en tiempo real
4. **Evaluación**: Revisa métricas finales y matriz de confusión
5. **Exportación**: El modelo se guarda en `checkpoints/`
6. **Inferencia**: Usa Streamlit o importa el modelo en código

## 👤 Autor

**Kevin Cajachuan**

- GitHub: [@Kajachuan](https://github.com/Kajachuan)