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
        chunks = split_docs(docs)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.from_documents(chunks, embeddings)
        st.session_state.db = db
        st.success("PDF successfully processed")

if st.session_state.db:
    question = st.text_input("Your Question:", placeholder="What is the main contribution of this journal?")

    if st.button("Answer"):
        if not question.strip():
            st.warning("Please enter your question.")
        else:
            with st.spinner("Please wait, ScholarBot is finding the answer..."):
                retriever = st.session_state.db.as_retriever()
                result = ask_question(question, retriever=retriever)
                st.markdown("### Answer:")
                st.write(result)

st.markdown("---")
st.info("ScholarBot is your assistant for answering questions related to scientific journals.")