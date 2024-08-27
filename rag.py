import streamlit as st

from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(model='llama3.1:latest', temperature=0)

chroma_local = Chroma(persist_directory="./vectordb2", embedding_function=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'))

# Función para crear el prompt
def prompt(texto):
    system_prompt = f"{texto}\n\n{{context}}"
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ])
    return prompt


# Texto del prompt inicial
texto_prompt = """
    "Usted es un asistente para tareas de respuesta a preguntas. Utiliza los siguientes elementos del contexto recuperado para responder 
    a la pregunta. Responde fielmente con la información pasada por el contexto.Si no conoce la respuesta, diga simplemente que necesitas
    para poder responder. Utiliza y procura que la respuesta sea concisa"
"""


def respuesta(pregunta, history):
    retriever = chroma_local.as_retriever()

    chain = create_stuff_documents_chain(llm, prompt(texto_prompt))
    rag = create_retrieval_chain(retriever, chain)
    
    results = rag.invoke({"input": pregunta})
    return results['answer']
  
  
  
  # Interfaz de usuario con Streamlit
def main():
    st.title("Asistente Virtual")
    
    # Mantener el historial de la conversación
    history = []
    
    # Caja de entrada para la pregunta
    pregunta = st.text_input("Haz una pregunta:")
    
    if st.button("Enviar"):
        if pregunta:
            respuesta_usuario = respuesta(pregunta, history)
            st.write("Respuesta:", respuesta_usuario)
            history.append((pregunta, respuesta_usuario))

if __name__ == "__main__":
    main()