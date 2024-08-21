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
    Tienes que responder fielmente con información de los documentos proporcionados. En caso de que la pregunta no te da el contexto necesario, 
    responde pidiendo lo que necesitas para poder contestarla. Te dejo algunas posibles preguntas que te pueden hacer 

1. **Presentación del Proyecto**
   - **Descripción del Negocio**: 
     - ¿Cuál es la idea básica del proyecto?
     - ¿Qué productos o servicios ofrecerá la empresa?
     - ¿Es el producto/servicio algo nuevo o una innovación sobre algo existente?
     - ¿Por qué decidiste crear esta empresa?
     - ¿Cuáles son los objetivos esperados a medio/largo plazo?

2. **Descripción de la Actividad de la Empresa**
   - **Descripción de la Actividad**:
     - ¿Cuál es la actividad principal de la empresa?
     - Describe más allá de su denominación, ¿qué se hará, para quién y cómo?
   - **Líneas de Negocio/Productos/Servicios**:
     - ¿Qué líneas de negocio, productos o servicios vas a ofrecer?
     - Describe cada línea de negocio o producto/servicio y su utilidad.

3. **Análisis del Mercado**
   - **Análisis de la Demanda**:
     - ¿Cuál es el mercado objetivo del proyecto?
     - ¿Dónde se comercializarán los productos o servicios?
     - ¿Quiénes son los clientes potenciales?
     - ¿Qué características tienen los clientes/consumidores potenciales?
     - ¿Qué necesidades han sido detectadas y cómo se cubrirán?
   - **Análisis de la Competencia**:
     - ¿Quiénes son los competidores directos e indirectos?
     - ¿Qué productos o servicios ofrecen y cómo se diferencian de tu oferta?
   - **Análisis de los Proveedores**:
     - ¿Quiénes serán tus proveedores principales y qué importancia tienen en tu actividad?
   - **Riesgos y Factores Claves de Éxito**:
     - ¿Cuáles son los principales riesgos que puede enfrentar tu empresa?
     - ¿Qué medidas planeas tomar para mitigar estos riesgos?
   - **Análisis DAFO**:
     - ¿Cuáles son las fortalezas y debilidades de tu proyecto?
     - ¿Qué oportunidades y amenazas existen en el sector?

4. **Marketing y Comercialización**
   - **Política de Producto**:
     - ¿Qué gama de productos o servicios ofrecerás?
     - ¿Cuáles son las principales líneas de negocio?
   - **Política de Precios**:
     - ¿Cuál será el precio medio de venta de tus productos o servicios?
     - ¿Cómo se compara con los precios de la competencia?
   - **Canales de Distribución**:
     - ¿Cómo se distribuirán los productos o servicios?
     - ¿Lo harás directamente o a través de distribuidores?
   - **Estrategia de Promoción**:
     - ¿Cómo darás a conocer tu producto o servicio?
     - ¿Qué medios utilizarás para la promoción?
   - **Plan de Ventas**:
     - ¿Cuál es la previsión de ventas para los próximos tres años?

5. **Producción y Operaciones**
   - **Infraestructuras e Instalaciones**:
     - ¿Dónde estará ubicada la empresa?
     - ¿Qué infraestructuras e instalaciones necesitas?
   - **Proceso de Fabricación/Prestación del Servicio**:
     - ¿Cuál es la capacidad productiva actual o esperada?
     - ¿Planeas implantar algún sistema de calidad?
   - **Aprovisionamiento y Logística**:
     - ¿Cuáles son las materias primas y otros suministros necesarios?
     - ¿Cómo se gestionarán las compras y la logística?

6. **Organización y Recursos Humanos**
   - **Organigrama**:
     - ¿Cómo estará estructurada la empresa?
     - ¿Cuántas personas estarán a cargo de cada departamento y cuáles serán sus responsabilidades?
   - **Equipo Directivo**:
     - ¿Quién formará parte del equipo directivo y cuál será su perfil profesional?
   - **Perfiles Profesionales**:
     - ¿Qué perfiles profesionales necesitas para operar la empresa?
     - ¿Cuántos puestos de trabajo requerirás para cada perfil?
   - **Retribución**:
     - ¿Cuál será la política de retribución?
     - ¿Salarios, incentivos, comisiones, etc.?
   - **Otras Políticas de Recursos Humanos**:
     - ¿Qué políticas de reclutamiento, selección, formación y motivación de personal seguirás?

7. **Inversión y Financiación**
   - **Inversión Inicial**:
     - ¿Cuál será la inversión inicial necesaria para poner en marcha la empresa?
     - ¿Por qué es necesaria esta inversión?
   - **Financiación Inicial**:
     - ¿Cuáles serán las fuentes de financiación, tanto propias como ajenas?
"""


def respuesta(pregunta, history):
    retriever = chroma_local.as_retriever()

    chain = create_stuff_documents_chain(llm, prompt(texto_prompt))
    rag = create_retrieval_chain(retriever, chain)
    
    results = rag.invoke({"input": pregunta})
    return results['answer']