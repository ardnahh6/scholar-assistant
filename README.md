# ScholarBot -- Your Scholar Assistant

ScholarBot is a Streamlit-based application designed to help researchers quickly extract and retrieve relevant information from academic papers (PDF format). It allows users to upload research papers, process them into vector embeddings, and query them using a natural language interface.

## 🚀 Features

- **PDF Upload**: Upload your academic papers in PDF format.
- **Text Splitting & Embedding**: Automatically splits document into chunks and embeds them using HuggingFace `all-MiniLM-L6-v2` model.
- **Vector Search with FAISS**: Stores embeddings locally in FAISS for fast retrieval.
- **Question Answering**: Ask questions and get answers based on the uploaded papers using Ollama `llama3.2` model.
- **Streamlit UI**: Simple and intuitive web interface for interaction.

## 🛠️ Tech Stack
- **[Python 3.11]**
- **[LangChain](https://www.langchain.com/)** – document loading, chunking, retrieval, and QA.
- **[FAISS](https://github.com/facebookresearch/faiss)** – vector store for efficient similarity search.
- **[HuggingFace Embeddings](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)** – embedding generation.
- **[Ollama](https://ollama.ai/)** – local LLM inference (`llama3.2`).
- **[Streamlit](https://streamlit.io/)** – web-based interface.

## ⚙️ Installation

1.  Clone the repository:
    ```     
    git clone https://github.com/ardnahh6/scholar-assistant.git     cd scholar-assistant
    ```

2.  Install dependencies: 
    ```    
    pip install -r requirements.txt
    ```

3.  Make sure you have **Ollama** installed and the desired model
    pulled, for example: 
    ```
    ollama pull llama3.2
    ```

## 🖥️ How to Use

1.  Start the Streamlit app: 
    ```
    streamlit run app/interface.py
    ```

2.  Upload your PDF(s) via the web interface.

3.  Ask questions in natural language.

4.  Receive concise answers with references to the uploaded document(s).