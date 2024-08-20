import streamlit as st
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Configuración del modelo y la base de datos de vectores
llm = ChatOllama(model='llama3.1:latest', temperature=0)
chroma_local = Chroma(persist_directory="./vectordb", embedding_function=HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2'))

# Función para crear el prompt
def prompt(texto):
    system_prompt = (
        texto +
        "\n\n" +
        "{context}"
    )

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ])
    return prompt_template

# Función para obtener la respuesta
def obtener_respuesta(pregunta, llm, chroma_db, prompt):
    retriever = chroma_db.as_retriever()
    chain = create_stuff_documents_chain(llm, prompt)
    rag = create_retrieval_chain(retriever, chain)
    
    results = rag.invoke({"input": pregunta})
    return results

# Texto del prompt inicial
texto_prompt = """Tú eres un asistente para tareas de respuesta a preguntas. 
Usa los siguientes fragmentos de contexto recuperado para responder 
la pregunta. Si no sabes la respuesta, di que no sabes. Mantén la respuesta concisa."""

# Aplicación de Streamlit
st.title("Asistente de Preguntas")

# Entrada de texto del usuario
pregunta = st.text_input("Escribe tu pregunta aquí:")

if st.button("Obtener Respuesta"):
    if pregunta:
        respuesta_obtenida = obtener_respuesta(pregunta, llm, chroma_local, prompt(texto_prompt))
        st.write("**Respuesta:**")
        st.write(respuesta_obtenida['answer'])
    else:
        st.write("Por favor, escribe una pregunta.")