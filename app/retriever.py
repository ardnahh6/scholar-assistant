import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

INDEX_PATH = "data/index"
MODEL_NAME = "llama3-8b-8192"

def load_groq_llm():
    print("[INFO] ⚡️ Using GroqCloud model:", MODEL_NAME)
    return ChatGroq(
        model_name=MODEL_NAME,
        temperature=0.5,
        groq_api_key=GROQ_API_KEY
    )

def ask_question(question):
    print("[INFO] 📚 Loading FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()

    llm = load_groq_llm()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)
    result = qa_chain.run(question)
    return result

if __name__ == "__main__":
    while True:
        q = input("❓ You > ")
        print("🤖 Bot >", ask_question(q))