import streamlit as st

from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(model='llama3.1:latest', temperature=0)
chroma_local = Chroma(persist_directory="./vectordb", embedding_function=HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2'))

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
    Eres un asistente virtual encargado de responder preguntas apoyándote en el contexto que tienes de los documentos pdf. 
    Tienes que responder fielmente con información de los documentos proporcionados. 
    En caso de que la pregunta no te da el contexto necesario, responde pidiendo lo que necesitas para poder contestarla.
    Simpre responde con la informacíon del contexto. No respondas que no tienes acceso a la informacíon.
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