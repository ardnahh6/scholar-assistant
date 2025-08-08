import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

INDEX_PATH = "data/index"

template = """
You are an academic research assistant. Answer the following questions completely and clearly based on the available documents.

Context:
{context}

Question:
{question}

Answer:
"""

# This is for testing in local before testing with streamlit

#def get_chain():
#    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#    db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
#    retriever = db.as_retriever()
#
#    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
#
#    model = OllamaLLM(model="llama3.2")
#
#    qa_chain = RetrievalQA.from_chain_type(
#        llm=model,
#        retriever=retriever,
#        return_source_documents=True,
#        chain_type_kwargs={"prompt": prompt}
#    )
#
#    return qa_chain


def ask_question(question: str, retriever=None):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if retriever is None:
        db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever()

    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm = OllamaLLM(model="llama3.2")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    result = qa_chain({"query": question})
    return result["result"]

if __name__ == "__main__":
    while True:
        q = input("❓ You > ")
        print("🤖 Bot >", ask_question(q))