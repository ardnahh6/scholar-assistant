import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
from transformers import AutoModelForCausalLM, AutoTokenizer, pipelines

load_dotenv()

INDEX_PATH = "data/index"
MODEL_NAME = "microsoft/phi-2"
HF_TOKEN = os.getenv("HF_TOKEN")

def load_model():
    print("[INFO] 🔃 Loading Phi-2 Model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HF_TOKEN)

    pipe = pipelines("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
    return HuggingFacePipeline(pipeline=pipe)

def ask_question(question):
    print("[INFO] 📚 Loading FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(question)

    llm = load_model()
    chain = load_qa_chain(llm, chain_type="stuff")
    result = chain.run(input_documents=docs, question=question)
    return result

if __name__ == "__main__":
    while True:
        q = input("❓ Anda > ")
        print("🤖 Bot >", ask_question(q))