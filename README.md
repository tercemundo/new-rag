# Chatbot con Groq y RAG

## Descripción General

Esta aplicación es un chatbot interactivo que utiliza los modelos de lenguaje de Groq (Llama y DeepSeek) para responder preguntas. La característica principal es su capacidad para consultar documentos PDF subidos por el usuario antes de recurrir al conocimiento general del modelo.

## ¿Qué es RAG?

RAG (Retrieval-Augmented Generation) o Generación Aumentada por Recuperación es una técnica que combina:

1. **Recuperación**: Busca información relevante en documentos o bases de datos.
2. **Generación**: Utiliza un modelo de lenguaje para generar respuestas basadas en la información recuperada.

Esta técnica mejora la precisión de las respuestas al proporcionar contexto específico al modelo, reduciendo las "alucinaciones" (información incorrecta) y permitiendo responder preguntas sobre datos que no estaban en el entrenamiento original del modelo.

## Funcionamiento de la Aplicación

### Bloques Principales

1. **Configuración Inicial**
   - Inicializa el estado de la sesión para almacenar mensajes, documentos y el historial de chat.
   - Configura la interfaz de usuario con Streamlit.

2. **Gestión de Documentos PDF**
   - Permite subir archivos PDF.
   - Procesa los PDFs dividiéndolos en fragmentos manejables.
   - Crea embeddings (representaciones vectoriales) de estos fragmentos para búsquedas eficientes.

3. **Sistema de Búsqueda Web**
   - Implementa una búsqueda en DuckDuckGo como respaldo.
   - Extrae contenido relevante de los resultados de búsqueda.

4. **Interfaz de Usuario**
   - Panel lateral para configuración (API key, modelo, temperatura).
   - Sección para subir y gestionar documentos.
   - Historial de chat y campo para preguntas.

5. **Procesamiento de Preguntas**
   - Sistema de tres niveles para responder preguntas.

### Lógica de Respuesta (Flujo de Trabajo)

La aplicación sigue un proceso de tres pasos para responder preguntas:

#### PASO 1: Búsqueda en Documentos (RAG)
- Si hay documentos cargados, primero intenta encontrar la respuesta en ellos.
- Utiliza el vectorstore FAISS para encontrar fragmentos relevantes.
- Evalúa si la respuesta encontrada es útil o si contiene frases de incertidumbre.
- Si encuentra una respuesta confiable, la muestra junto con las fuentes.

#### PASO 2: Consulta al Modelo de Lenguaje
- Si no hay documentos o la respuesta del RAG es incierta, consulta directamente al modelo.
- Utiliza un mensaje de sistema que anima al modelo a usar su conocimiento general.
- Siempre responde en español por defecto.
- Muestra la respuesta de forma progresiva (streaming).

#### PASO 3: Búsqueda Web (Respaldo)
- Si el modelo indica incertidumbre y la búsqueda web está habilitada, busca en internet.
- Extrae contenido relevante de los resultados de búsqueda.
- Genera una respuesta basada en la información web encontrada.
- Incluye las fuentes web en la respuesta.

## Características Destacadas

- **Respuestas Multimodales**: Combina conocimiento de documentos, modelo y web.
- **Interfaz Intuitiva**: Fácil de usar con Streamlit.
- **Respuestas en Español**: Configurado para responder en español por defecto.
- **Citación de Fuentes**: Indica de dónde proviene la información (páginas de PDF o URLs).
- **Manejo Inteligente de Incertidumbre**: Si no encuentra información en los documentos, busca alternativas automáticamente.

## Requisitos

- API Key de Groq
- Python con las bibliotecas necesarias (streamlit, groq, langchain, etc.)
- Conexión a internet para búsquedas web y consultas a la API