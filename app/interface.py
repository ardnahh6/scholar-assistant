import os
import streamlit as st
import tempfile
from retriever import ask_question
from ingestion import load_pdfs, split_docs
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="ScholarBot", page_icon="📄")
st.title("📄 ScholarBot -- Your Scholar Assistant")
st.markdown("Upload your PDF's Journal file and ask anything:))")

if "db" not in st.session_state:
    st.session_state.db = None

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with st.spinner("Processing PDF..."):
        docs = load_pdfs(os.path.dirname(tmp_path))