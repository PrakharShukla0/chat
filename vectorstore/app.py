import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import os

st.title("📚 Medical Chatbot")

# Load the vector store
db = FAISS.load_local("vectorstore/db_faiss", HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))

# LLM setup
llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature":0.5, "max_length":1000}, huggingfacehub_api_token=os.getenv("HF_TOKEN"))

# Retrieval QA chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

query = st.text_input("Ask me anything from the medical doc:")

if query:
    with st.spinner("Thinking..."):
        result = qa.run(query)
        st.write(result)