<!-- Logo o imagen del proyecto -->
<p align="center">
  <img src="resources\img_rag.png" alt="Logo del Proyecto" width="800">
</p>

# Retrieval-Augmented Generation (RAG)

### Description

In this repo, I have created a new project named Retrieval-Augmented Generation (RAG). This project focuses on enhancing the capabilities of language models by integrating information retrieval techniques, enabling more accurate and contextually relevant responses by sourcing information from external documents or databases.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Resources](#Resources)

## Installation

Follow these steps to set up the development environment on your local machine:

  Clone the repository:

    git clone https://github.com/pablopuch/rag.git
    cd rag


  Create and activate a virtual environment:

    python -m venv env
    source env/bin/activate # macOS/Linux
    .\env\Scripts\activate # Windows


  Install the necessary dependencies:
  
    pip install -r requirements.txt


  Create de file doc and put your pdfs in this file.

    mkdir doc

  Run ollama serve

    ollama run llama3.1:latest

  Initialize with the command

    py app.py

    

## Usage



## Project Structure

<p align="center">
  <img src="resources\scheme.jpg" alt="Logo del Proyecto" width="800">
</p>

## Documentation



## Resources

https://www.youtube.com/watch?v=ApZvYZIwSeE&list=PLCwl8iPaU6OLkMwhpsKmKxgb9kZqWZyYh&pp=iAQB

https://github.com/fcori47/rag_basico/blob/master/Clase%202%20-%20VectorDB.ipynb

https://medium.com/the-ai-forum/rag-on-complex-pdf-using-llamaparse-langchain-and-groq-5b132bd1f9f3

https://www.youtube.com/watch?v=Mg3xOWWaF0c&t=999s

https://python.langchain.com/v0.2/docs/how_to/semantic-chunker/

https://medium.com/the-ai-forum/semantic-chunking-for-rag-f4733025d5f5

https://huggingface.co/spaces/Nymbo/chunk_visualizer


