# Sistema RAG para Recomendación de Películas

Un sistema completo de Retrieval-Augmented Generation (RAG) que utiliza embeddings de texto para responder preguntas sobre películas basándose en una base de conocimientos de sinopsis cinematográficas. Incluye APIs REST para integración web.

## 📋 Descripción

Este proyecto implementa un sistema RAG completo que:

- **Preprocesa** texto de sinopsis de películas con pipeline optimizado
- **Genera embeddings** usando múltiples modelos (FlagEmbedding, SentenceTransformer, OpenAI)
- **Indexa** los embeddings en archivos Parquet para búsqueda eficiente
- **Responde preguntas** combinando búsqueda por similitud y generación de texto con GPT-4
- **Proporciona APIs REST** para integración web y servicios

## 🏗️ Arquitectura del Proyecto

```
project_1/
├── config.py                 # Configuración de modelos y parámetros
├── preprocessing.py          # Pipeline de preprocesamiento de texto
├── embeddings_factory.py     # Factory para diferentes tipos de embeddings
├── embeddings_indexing.py    # Script para crear la base de conocimientos
├── rag.py                   # Sistema RAG principal
├── flask_api.py             # API Flask completa con endpoints
├── api.py                   # API Flask simplificada
├── rag_model.pickle         # Modelo RAG serializado (143MB)
├── data/                    # Datos y embeddings preprocesados
│   ├── movies-dataset.csv
│   ├── movies-dataset.parquet
│   ├── movies-dataset-embeddings-flag.parquet
│   ├── movies-dataset-embeddings-sentence_transformer.parquet
│   └── movies-dataset-embeddings-openai.parquet
├── requirements.txt         # Dependencias del proyecto
└── README.md               # Documentación
```
<img width="1080" height="1512" alt="image" src="https://github.com/user-attachments/assets/905da908-3e42-4aca-9fbc-a557b056162c" />



## 🚀 Instalación

1. **Clonar el repositorio:**
   ```bash
   git clone <repository-url>
   cd project_1
   ```

2. **Crear entorno virtual:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # En Windows: .venv\Scripts\activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Datos

El sistema utiliza un dataset de películas que incluye:
- **movies-dataset.csv**: Dataset original con información de películas (5.9MB)
- **movies-dataset.parquet**: Versión optimizada en formato Parquet (3.6MB)
- **Archivos de embeddings**: Base de conocimientos con embeddings precalculados
  - FlagEmbedding: 69KB (373 líneas)
  - SentenceTransformer: 11MB
  - OpenAI: 151KB (723 líneas)

### Formatos de Embeddings Soportados:
- **FlagEmbedding**: BAAI/bge-m3 (alto rendimiento, menor tamaño)
- **SentenceTransformer**: BAAI/bge-small-en-v1.5 (equilibrio rendimiento/velocidad)
- **OpenAI**: text-embedding-3-small (requiere API key)

## 🔧 Uso

### 1. Crear Base de Conocimientos

Para generar embeddings y crear la base de conocimientos:

```bash
python embeddings_indexing.py
```

Esto creará archivos Parquet con embeddings para cada tipo de modelo configurado.

### 2. Ejecutar Sistema RAG

```bash
python rag.py
```

El sistema cargará automáticamente:
- La base de conocimientos según la configuración en `config.py`
- El preprocesador de texto
- El modelo de embeddings especificado
- Guardará el modelo serializado en `rag_model.pickle`

### 3. API REST

#### API Completa (flask_api.py)
```bash
python flask_api.py
```

**Endpoints disponibles:**
- `GET /` - Información de la API
- `GET /health` - Verificar estado de la API
- `GET /query?question=<pregunta>` - Consulta simple
- `POST /query` - Consulta con JSON

**Ejemplo de uso:**
```bash
# GET request
curl "http://localhost:5000/query?question=What is the name of the movie where humans and AIs coexist?"

# POST request
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the name of the movie where humans and AIs coexist?"}'
```

#### API Simplificada (api.py)
```bash
python api.py
```

**Endpoints:**
- `GET /health` - Estado de la API
- `GET /query` - Consulta (requiere JSON con campo 'question')

### 4. Configuración

Edita `config.py` para personalizar:

```python
# Tipo de embeddings a usar
embedding_to_use = "sentence_transformer"

# Umbral de similitud para filtrar resultados
similarity_threshold = 0.55

# Modelos disponibles
embedding_models = {
    "flag": {"model": "BAAI/bge-m3"},
    "sentence_transformer": {"model": "BAAI/bge-small-en-v1.5"},
    "openai": {"model": "text-embedding-3-small"}
}
```

## 🧠 Componentes Principales

### TextPreprocessor
Pipeline de preprocesamiento optimizado que incluye:
- Conversión a minúsculas
- Eliminación de acentos y caracteres especiales
- Limpieza de secuencias de escape
- Eliminación de emojis
- Eliminación de puntuación
- Normalización de espacios
- Procesamiento por lotes

### EmbeddingFactory
Factory pattern que soporta múltiples tipos de embeddings:
- **FlagEmbeddingGenerator**: Para modelos BGE-M3 (optimizado con FP16)
- **SentenceTransformerGenerator**: Para modelos SentenceTransformer
- **OpenAIEmbeddingGenerator**: Para embeddings de OpenAI API
- **Generación por lotes**: Para procesamiento eficiente de múltiples textos

### RAG System
Sistema principal que:
1. Preprocesa la consulta del usuario
2. Genera embeddings para la consulta
3. Calcula similitud coseno con la base de conocimientos
4. Transforma distancias a similitud (1 - distancia)
5. Filtra por umbral de similitud configurable
6. Genera contexto relevante con URLs de imágenes
7. Usa GPT-4 para generar respuesta final
8. Serializa el modelo para uso posterior

### APIs Flask
- **flask_api.py**: API completa con manejo de errores, validación y documentación
- **api.py**: API simplificada para casos de uso básicos
- **Endpoints de salud**: Para monitoreo y verificación de estado
- **Manejo de errores**: Respuestas HTTP apropiadas con códigos de estado

## ⚙️ Configuración Avanzada

### Cambiar Modelo de Embeddings

```python
# En config.py
embedding_to_use = "flag"  # o "sentence_transformer" o "openai"
```

### Ajustar Umbral de Similitud

```python
# En config.py
similarity_threshold = 0.80  # Más estricto (actual: 0.55)
```

### Personalizar Prompt

Modifica la variable `prompt` en `config.py` para cambiar el comportamiento del sistema RAG.

### Configuración de OpenAI

El sistema incluye una API key de OpenAI configurada para GPT-4. Para usar tu propia key:

```python
# En rag.py, línea 58
os.environ["OPENAI_API_KEY"] = "tu-api-key-aqui"
```

## 🔍 Ejemplo de Uso

### Uso Directo del Sistema RAG

```python
from rag import RAG
from preprocessing import TextPreprocessor
import pandas as pd

# Cargar base de conocimientos
knowledge_base = pd.read_parquet("data/movies-dataset-embeddings-sentence_transformer.parquet")

# Inicializar sistema
preprocessor = TextPreprocessor()
rag = RAG(knowledge_base, preprocessor, embedding_type='sentence_transformer')

# Hacer consulta
question = "The film begins in a diner where Peter Parker"
response, context = rag.query(question)

print(f"Pregunta: {question}")
print(f"Respuesta: {response}")
print(f"Contexto: {context}")
```

### Uso con Modelo Serializado

```python
import pickle

# Cargar modelo pre-entrenado
with open('rag_model.pickle', 'rb') as f:
    rag = pickle.load(f)

# Hacer consulta
response, context = rag.query("What is the movie about?")
print(response)
```

## 📈 Rendimiento y Características

### Modelos de Embeddings
- **FlagEmbedding**: Mayor precisión, menor tamaño de archivo (69KB)
- **SentenceTransformer**: Equilibrio entre velocidad y precisión (11MB)
- **OpenAI**: Requiere conexión a internet, costo por uso (151KB)

### Optimizaciones
- **FP16**: Uso de precisión media para acelerar inferencia
- **Parquet**: Formato optimizado para almacenamiento y lectura
- **Serialización**: Modelo guardado en pickle para carga rápida
- **Procesamiento por lotes**: Para generación eficiente de embeddings

### APIs
- **Flask**: Framework web ligero y rápido
- **JSON**: Formato de intercambio de datos
- **CORS**: Configurado para integración web
- **Health checks**: Para monitoreo de servicios

## 🛠️ Requisitos del Sistema

### Dependencias Principales
- Python 3.8+
- pandas>=2.0.0
- numpy>=1.24.0
- scipy>=1.10.0
- sentence-transformers>=2.2.0
- FlagEmbedding>=1.2.0
- openai>=1.0.0
- flask>=2.3.0

### Requisitos de Hardware
- 8GB+ RAM (recomendado para modelos grandes)
- Conexión a internet (para descargar modelos y usar OpenAI)
- 500MB+ espacio en disco (para modelos y embeddings)

### Dependencias Opcionales
- torch>=2.0.0 (para mejor rendimiento)
- transformers>=4.30.0
- pytest>=7.0.0 (para testing)
- black>=23.0.0 (para formateo de código)

## 🔧 Desarrollo

### Estructura de Código
- **Patrón Factory**: Para generación de embeddings
- **Patrón Strategy**: Para diferentes tipos de preprocesamiento
- **Separación de responsabilidades**: Cada componente tiene una función específica
- **Configuración centralizada**: Todos los parámetros en `config.py`

### Testing
```bash
# Instalar dependencias de desarrollo
pip install pytest black flake8

# Ejecutar tests
pytest

# Formatear código
black .

# Linting
flake8
```

## 📝 Notas de Implementación

- El sistema utiliza **similitud coseno** para calcular la relevancia
- Las **distancias se transforman** a similitud (1 - distancia) para facilitar interpretación
- El **umbral de similitud** es configurable (actual: 0.55)
- Se **filtran los top 2 resultados** más relevantes para generar contexto
- El **prompt incluye URLs de imágenes** cuando están disponibles
- Las **respuestas se generan en inglés** según la configuración

## 🚀 Próximos Pasos

- [ ] Implementar cache de embeddings para consultas repetidas
- [ ] Añadir autenticación a las APIs
- [ ] Implementar rate limiting
- [ ] Añadir métricas de rendimiento
- [ ] Crear interfaz web
- [ ] Implementar búsqueda por filtros adicionales
- [ ] Añadir soporte para múltiples idiomas

