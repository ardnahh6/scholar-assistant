import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_PATH = "data/papers"
INDEX_PATH = "data/index"

def load_pdfs(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            loader = PyMuPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
    return documents

def split_docs(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)

def ingest():
    print("Loading PDF Files...")
    documents = load_pdfs(DATA_PATH)
    print(f"{len(documents)} raw documents loaded.")

    print("Splitting documents...")
    chunks = split_docs(documents)
    print(f"{len(chunks)} chunks created.")

    print("Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print("Saving to FAISS index...")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(INDEX_PATH)
    print("Ingestion complete.")

if __name__ == "__main__":
    ingest()