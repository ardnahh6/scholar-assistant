from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import retrieval_qa
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import huggingface_hub

load_dotenv()

INDEX_PATH = "data/index"

prompt = """
You are an academic research assistant. Answer the following questions completely and clearly based on the available documents.

Question:
{question}

Answer:
"""

