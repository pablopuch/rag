import gradio as gr
from common.rag import respuesta

gr.ChatInterface(respuesta).launch()