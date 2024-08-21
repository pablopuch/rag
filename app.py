import gradio as gr
from common.rag import obtener_respuesta

gr.ChatInterface(obtener_respuesta).launch()