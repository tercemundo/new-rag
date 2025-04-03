import streamlit as st
from groq import Groq
import os
import tempfile
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
import requests
from bs4 import BeautifulSoup
import uuid

# Configuración de la página
st.set_page_config(
    page_title="Chatbot Groq",
    page_icon="🤖",
    layout="wide"
)

# Inicializar el estado de la sesión para el historial de chat si no existe
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_pdfs" not in st.session_state:
    st.session_state.uploaded_pdfs = {}

# Título y descripción de la aplicación
st.title("🤖 Chatbot Groq")
st.markdown("Chatea con modelos Llama usando la API de Groq y tus documentos PDF")

# Función para obtener la clave API de secrets o entrada de usuario
def obtener_api_key():
    # Intentar obtener la clave API de los secretos de Streamlit
    try:
        return st.secrets["GROQ_API_KEY"]
    except:
        # Si no está en secretos, obtener del estado de la sesión o entrada de usuario
        if "api_key" not in st.session_state or not st.session_state.api_key:
            st.session_state.api_key = ""
        
        api_key = st.sidebar.text_input("Ingresa tu clave API de Groq:", type="password", value=st.session_state.api_key)
        if api_key:
            st.session_state.api_key = api_key
        return api_key

# Función para procesar PDF subido
def procesar_pdf(archivo_subido):
    # Crear un archivo temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as archivo_tmp:
        # Escribir los datos del archivo subido al archivo temporal
        archivo_tmp.write(archivo_subido.getvalue())
        ruta_tmp = archivo_tmp.name
    
    try:
        # Cargar PDF
        cargador = PyPDFLoader(ruta_tmp)
        documentos = cargador.load()
        
        # Añadir nombre de archivo a los metadatos
        for doc in documentos:
            doc.metadata["source"] = archivo_subido.name
        
        # Dividir documentos en fragmentos
        divisor_texto = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        fragmentos_documento = divisor_texto.split_documents(documentos)
        
        # Almacenar información del PDF
        pdf_id = str(uuid.uuid4())
        st.session_state.uploaded_pdfs[pdf_id] = {
            "name": archivo_subido.name,
            "chunks": fragmentos_documento
        }
        
        # Actualizar todos los fragmentos de documentos
        todos_fragmentos = []
        for info_pdf in st.session_state.uploaded_pdfs.values():
            todos_fragmentos.extend(info_pdf["chunks"])
        
        st.session_state.document_chunks = todos_fragmentos
        
        # Crear embeddings y vectorstore
        try:
            try:
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
            except (RuntimeError, ConnectionError, requests.exceptions.ConnectionError):
                # Primer fallback: configuración más simple
                try:
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                        cache_folder="./embeddings_cache"
                    )
                except (RuntimeError, ConnectionError, requests.exceptions.ConnectionError):
                    # Segundo fallback: usar un modelo local si está disponible
                    local_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings_cache")
                    if os.path.exists(local_model_path):
                        embeddings = HuggingFaceEmbeddings(
                            model_name=local_model_path,
                            model_kwargs={'device': 'cpu'}
                        )
                    else:
                        st.warning("No se puede conectar a HuggingFace. Usando un método alternativo para procesar el texto.")
                        # Implementar un método simple de embeddings como fallback
                        from sklearn.feature_extraction.text import TfidfVectorizer
                        
                        class SimpleEmbeddings:
                            def __init__(self):
                                self.vectorizer = TfidfVectorizer()
                                self.fitted = False
                                
                            def embed_documents(self, texts):
                                if not self.fitted:
                                    self.vectorizer.fit(texts)
                                    self.fitted = True
                                return self.vectorizer.transform(texts).toarray()
                        
                        embeddings = SimpleEmbeddings()
            
            vectorstore = FAISS.from_documents(todos_fragmentos, embeddings)
            st.session_state.vectorstore = vectorstore
            
            return len(fragmentos_documento), archivo_subido.name
        except Exception as e:
            st.error(f"Error al procesar PDF: {str(e)}")
            return 0, archivo_subido.name
    finally:
        # Limpiar el archivo temporal
        os.unlink(ruta_tmp)

# Función para buscar en la web
def buscar_web(consulta, num_resultados=5):
    try:
        # Añadir "actual" o "actual" a consultas sensibles al tiempo para obtener resultados más recientes
        # Mejorando la búsqueda temporal en app6.py
        
        # Voy a modificar la función de búsqueda web para que maneje mejor las consultas temporales, incluyendo referencias a años específicos desde 2023 hasta 2025.
        # Añadir "actual" o año específico a consultas sensibles al tiempo para obtener resultados más recientes
        años_mencionados = [str(año) for año in range(2023, 2026) if str(año) in consulta.lower()]
        
        if años_mencionados:
            # Si ya hay un año específico en la consulta, usarlo tal cual
            consulta = f"{consulta}"
        elif any(palabra in consulta.lower() for palabra in ["presidente", "actual", "ahora", "hoy", "quien es"]):
            # Para consultas sobre información actual sin año específico, añadir el año actual
            from datetime import datetime
            año_actual = datetime.now().year
            consulta = f"actual {consulta} {año_actual}"
        
        # Usar búsqueda DuckDuckGo
        url_busqueda = f"https://lite.duckduckgo.com/lite/?q={consulta}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        respuesta = requests.get(url_busqueda, headers=headers)
        soup = BeautifulSoup(respuesta.text, 'html.parser')
        
        # Extraer resultados de búsqueda
        resultados = []
        for etiqueta_a in soup.find_all('a', href=True):
            if etiqueta_a['href'].startswith('https://') and not etiqueta_a['href'].startswith('https://duckduckgo.com'):
                resultados.append({
                    "title": etiqueta_a.get_text().strip(),
                    "url": etiqueta_a['href']
                })
                if len(resultados) >= num_resultados:
                    break
        
        # Obtener contenido de múltiples resultados en lugar de solo el primero
        contenido = ""
        for resultado in resultados[:2]:  # Intentar obtener contenido de los 2 primeros resultados
            try:
                respuesta_contenido = requests.get(resultado["url"], headers=headers, timeout=5)
                soup_contenido = BeautifulSoup(respuesta_contenido.text, 'html.parser')
                
                # Extraer texto de párrafos
                parrafos = soup_contenido.find_all('p')
                contenido_resultado = "\n".join([p.get_text().strip() for p in parrafos[:8]])  # Obtener más párrafos
                
                if contenido_resultado:
                    contenido += f"\nDe {resultado['url']}:\n{contenido_resultado}\n\n"
                    
                    # Si tenemos suficiente contenido, detener
                    if len(contenido) > 2000:
                        break
            except:
                continue
        
        return {
            "success": True,
            "results": resultados,
            "content": contenido or "No se pudo recuperar contenido de las páginas web."
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error al buscar en la web: {str(e)}"
        }

# Barra lateral para configuración
with st.sidebar:
    st.header("Configuración")
    
    # Obtener clave API
    api_key = obtener_api_key()
    
    # Selección de modelo
    modelo = st.selectbox(
        "Seleccionar Modelo:",
        [
            "llama3-8b-8192",
            "llama3-70b-8192",
            "llama2-70b-4096"
        ]
    )
    
    # Control deslizante de temperatura
    temperatura = st.slider("Temperatura:", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    
    # Sección de carga de PDF
    st.header("Subir Documentos")
    archivo_subido = st.file_uploader("Subir PDFs", type="pdf", accept_multiple_files=True)
    
    if archivo_subido:
        if st.button("Procesar PDFs"):
            with st.spinner("Procesando PDFs..."):
                total_fragmentos = 0
                for pdf in archivo_subido:
                    fragmentos, nombre_archivo = procesar_pdf(pdf)
                    if fragmentos > 0:
                        total_fragmentos += fragmentos
                        st.success(f"✅ {nombre_archivo}: {fragmentos} fragmentos")
                    else:
                        st.error(f"❌ Error al procesar {nombre_archivo}")
                
                if total_fragmentos > 0:
                    st.success(f"¡Todos los PDFs procesados! Total de fragmentos: {total_fragmentos}")
    
    # Mostrar PDFs subidos
    if st.session_state.uploaded_pdfs:
        st.subheader("PDFs Subidos")
        for pdf_id, info_pdf in st.session_state.uploaded_pdfs.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"📄 {info_pdf['name']} ({len(info_pdf['chunks'])} fragmentos)")
            with col2:
                if st.button("Eliminar", key=f"remove_{pdf_id}"):
                    del st.session_state.uploaded_pdfs[pdf_id]
                    # Actualizar todos los fragmentos de documentos
                    todos_fragmentos = []
                    for pdf_restante in st.session_state.uploaded_pdfs.values():
                        todos_fragmentos.extend(pdf_restante["chunks"])
                    
                    st.session_state.document_chunks = todos_fragmentos
                    
                    # Recrear vectorstore si todavía hay documentos
                    if todos_fragmentos:
                        try:
                            try:
                                embeddings = HuggingFaceEmbeddings(
                                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                                    model_kwargs={'device': 'cpu'},
                                    encode_kwargs={'normalize_embeddings': True}
                                )
                            except (RuntimeError, ConnectionError, requests.exceptions.ConnectionError):
                                # Primer fallback: configuración más simple
                                try:
                                    embeddings = HuggingFaceEmbeddings(
                                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                                        cache_folder="./embeddings_cache"
                                    )
                                except (RuntimeError, ConnectionError, requests.exceptions.ConnectionError):
                                    # Segundo fallback: usar un modelo local si está disponible
                                    local_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings_cache")
                                    if os.path.exists(local_model_path):
                                        embeddings = HuggingFaceEmbeddings(
                                            model_name=local_model_path,
                                            model_kwargs={'device': 'cpu'}
                                        )
                                    else:
                                        st.warning("No se puede conectar a HuggingFace. Usando un método alternativo para procesar el texto.")
                                        # Implementar un método simple de embeddings como fallback
                                        from sklearn.feature_extraction.text import TfidfVectorizer
                                        
                                        class SimpleEmbeddings:
                                            def __init__(self):
                                                self.vectorizer = TfidfVectorizer()
                                                self.fitted = False
                                            
                                            def embed_documents(self, texts):
                                                if not self.fitted:
                                                    self.vectorizer.fit(texts)
                                                    self.fitted = True
                                                return self.vectorizer.transform(texts).toarray()
                                        
                                        embeddings = SimpleEmbeddings()
                            
                            st.session_state.vectorstore = FAISS.from_documents(todos_fragmentos, embeddings)
                        except Exception as e:
                            st.error(f"Error al recrear vectorstore: {str(e)}")
                            st.session_state.vectorstore = None
                    else:
                        st.session_state.vectorstore = None
                    
                    st.rerun()
    
    # Opciones de búsqueda
    st.header("Opciones de Búsqueda")
    habilitar_busqueda_web = st.checkbox("Habilitar búsqueda web como respaldo", value=True)
    
    # Botón para limpiar chat
    if st.button("Limpiar Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()
    
    # Botón para limpiar todos los documentos
    if st.button("Limpiar Todos los Documentos"):
        st.session_state.vectorstore = None
        st.session_state.document_chunks = []
        st.session_state.uploaded_pdfs = {}
        st.success("¡Todos los documentos eliminados!")
        st.rerun()

# Mostrar mensajes de chat
for mensaje in st.session_state.messages:
    with st.chat_message(mensaje["role"]):
        st.markdown(mensaje["content"])

# Entrada de chat
if prompt := st.chat_input("Pregunta algo..."):
    # Verificar si se proporciona la clave API
    if not api_key:
        st.error("Por favor, ingresa tu clave API de Groq en la barra lateral o configúrala en los secretos de Streamlit.")
        st.stop()
    
    # Añadir mensaje del usuario al historial de chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Mostrar mensaje del usuario
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Mostrar respuesta del asistente con un spinner
    with st.chat_message("assistant"):
        marcador_mensaje = st.empty()
        respuesta_completa = ""
        
        try:
            # PASO 1: Intentar encontrar respuesta en PDFs si hay documentos disponibles
            respuesta_pdf = None
            fuentes_pdf = None
            tiene_contenido_pdf = False
            modelo_es_incierto = False  # Inicializar esta variable aquí
            
            # Detectar si la pregunta es sobre eventos recientes o información actual
            es_pregunta_temporal = any(palabra in prompt.lower() for palabra in [
                "2024", "2025", "actual", "reciente", "último", "ultima", 
                "ganó", "gano", "ganador", "campeón", "campeon", "presidente actual",
                "ahora", "este año", "este mes", "esta semana", "hoy"
            ])
            
            if st.session_state.vectorstore is not None:
                with st.spinner("Buscando en tus documentos..."):
                    # Inicializar LangChain con Groq
                    llm = ChatGroq(
                        groq_api_key=api_key,
                        model_name=modelo,
                        temperature=temperatura
                    )
                    
                    # Crear cadena de recuperación con prompt personalizado
                    recuperador = st.session_state.vectorstore.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 5}
                    )
                    
                    qa_chain = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        retriever=recuperador,
                        return_source_documents=True,
                        verbose=True
                    )
                    
                    # Obtener respuesta de RAG
                    resultado = qa_chain({"question": prompt, "chat_history": st.session_state.chat_history})
                    respuesta_pdf = resultado["answer"]
                    tiene_contenido_pdf = True
                    
                    # Guardar información de fuente para uso posterior incluso si la respuesta es incierta
                    if "source_documents" in resultado and resultado["source_documents"]:
                        fuentes = {}
                        for doc in resultado["source_documents"]:
                            if hasattr(doc, 'metadata'):
                                fuente = doc.metadata.get('source', 'Desconocido')
                                pagina = doc.metadata.get('page', 'Página desconocida')
                                if fuente not in fuentes:
                                    fuentes[fuente] = set()
                                fuentes[fuente].add(f"Página {pagina}")
                        
                        if fuentes:
                            texto_fuente = "\n\n**Fuentes:**\n"
                            for fuente, paginas in fuentes.items():
                                str_paginas = ", ".join(sorted(paginas))
                                texto_fuente += f"- {fuente} ({str_paginas})\n"
                            fuentes_pdf = texto_fuente
                    
                    # Verificar si la respuesta es útil (no solo "No lo sé" o similar)
                    frases_inciertas = [
                        "i don't know", "i don't have", "i cannot", "i can't", 
                        "no information", "not mentioned", "not specified",
                        "not provided", "no context", "no data", "cannot answer",
                        "unable to provide", "don't have enough", "no specific",
                        "no details", "not clear", "not available", "lo siento",
                        "no tengo información", "la información que tengo", "no incluye información",
                        "no tengo esa información", "no se encuentra", "no se menciona",
                        "no menciona", "no proporciona", "no habla sobre", "no contiene"
                    ]
                    
                    # Verificar si la respuesta contiene frases de incertidumbre
                    contiene_incertidumbre = any(frase in respuesta_pdf.lower() for frase in frases_inciertas)
                    
                    # También verificar si la respuesta es realmente relevante para la pregunta
                    # Esto ayuda a detectar casos donde el modelo encontró contenido pero no es relevante
                    verificacion_relevancia = True
                    
                    # Si la respuesta menciona que la información es sobre otra cosa
                    if any(frase in respuesta_pdf.lower() for frase in [
                        "la información que tengo es sobre", 
                        "ya que la información", 
                        "no tengo esa información en el contexto",
                        "la información proporcionada no",
                        "el texto proporcionado no",
                        "el texto solo habla sobre",
                        "el documento no menciona"
                    ]):
                        verificacion_relevancia = False
                    
                    # Si es una pregunta sobre eventos recientes y hay incertidumbre, ir directamente a la web
                    if es_pregunta_temporal and (contiene_incertidumbre or not verificacion_relevancia):
                        # Marcar para ir directamente a la búsqueda web
                        respuesta_completa = ""
                        modelo_es_incierto = True
                    # Solo usar respuesta PDF si no contiene frases de incertidumbre y es relevante
                    elif not contiene_incertidumbre and verificacion_relevancia:
                        # Actualizar historial de chat para contexto RAG
                        st.session_state.chat_history.append((prompt, respuesta_pdf))
                        
                        # Añadir información de fuentes
                        if fuentes_pdf:
                            respuesta_pdf += fuentes_pdf
                        
                        respuesta_completa = respuesta_pdf
                        marcador_mensaje.markdown(respuesta_completa)
                    # Si es incierto o no relevante, pasaremos al paso LLM (no establecer respuesta_completa)
                    else:
                        # No mostrar la respuesta incierta al usuario
                        # Simplemente pasar silenciosamente al paso LLM
                        pass
            
            # PASO 2: Si no hay buena respuesta de PDFs o no hay PDFs en absoluto, usar el modelo directamente
            # Saltamos este paso para preguntas temporales si el PDF no tuvo respuesta
            if respuesta_completa == "" and not (es_pregunta_temporal and modelo_es_incierto):
                with st.spinner("Pensando con LLM..."):
                    # Usar el modelo directamente para responder
                    cliente = Groq(api_key=api_key)
                    respuesta_llm = cliente.chat.completions.create(
                        model=modelo,
                        messages=[
                            {"role": "system", "content": "Eres un asistente útil y amigable. Responde de manera clara y concisa."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=temperatura
                    )
                    
                    respuesta_completa = respuesta_llm.choices[0].message.content
                    marcador_mensaje.markdown(respuesta_completa)
                    
                    # Actualizar historial de chat para contexto futuro
                    st.session_state.chat_history.append((prompt, respuesta_completa))
            
            # PASO 3: Si el modelo no sabe y la búsqueda web está habilitada, intentar búsqueda web
            if habilitar_busqueda_web and (respuesta_completa == "" or modelo_es_incierto):
                with st.spinner("Buscando en la web..."):
                    try:
                        # Si es una pregunta temporal, añadir el año actual a la consulta
                        consulta_web = prompt
                        if es_pregunta_temporal and "2024" not in prompt.lower():
                            from datetime import datetime
                            año_actual = datetime.now().year
                            consulta_web = f"{prompt} {año_actual}"
                        
                        resultados_web = buscar_web(consulta_web)
                        
                        if resultados_web["success"] and resultados_web["results"]:
                            # Formatear resultados web
                            contenido_web = resultados_web.get("content", "")
                            
                            # Usar modelo para generar respuesta basada en contenido web
                            cliente = Groq(api_key=api_key)
                            prompt_web = f"""Basado en la siguiente información de la web, por favor responde a la pregunta: "{prompt}"
                            
IMPORTANTE: Si la pregunta es sobre información actual (como quién es el presidente actual, eventos recientes, etc.), asegúrate de proporcionar la información MÁS RECIENTE disponible en el contenido web. Prioriza la información de 2024 sobre información más antigua.

Contenido web:
{contenido_web}

Resultados de búsqueda web:
"""
                            for i, resultado in enumerate(resultados_web["results"][:3], 1):
                                prompt_web += f"{i}. {resultado['title']} - {resultado['url']}\n"
                            
                            respuesta_web = cliente.chat.completions.create(
                                model=modelo,
                                messages=[
                                    {"role": "system", "content": "Eres un asistente útil que proporciona información precisa basada en los resultados de búsqueda web. Cita tus fuentes."},
                                    {"role": "user", "content": prompt_web}
                                ],
                                temperature=0.3  # Temperatura más baja para respuestas más precisas
                            )
                            
                            respuesta_completa = respuesta_web.choices[0].message.content
                            respuesta_completa += "\n\n*Respuesta generada con información de la web.*"
                            marcador_mensaje.markdown(respuesta_completa)
                            
                            # Actualizar historial de chat
                            st.session_state.chat_history.append((prompt, respuesta_completa))
                        else:
                            # Si la búsqueda web falló y aún no tenemos respuesta, mostrar mensaje de error
                            if respuesta_completa == "":
                                respuesta_completa = "Lo siento, no pude encontrar información relevante para responder tu pregunta. Por favor, intenta reformularla o pregunta algo diferente."
                                marcador_mensaje.markdown(respuesta_completa)
                    except Exception as e:
                        st.error(f"Error al buscar en la web: {str(e)}")
                        if respuesta_completa == "":
                            respuesta_completa = "Lo siento, ocurrió un error al buscar información en la web. Por favor, intenta nuevamente más tarde."
                            marcador_mensaje.markdown(respuesta_completa)