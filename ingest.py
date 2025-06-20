from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

DATA_DIR = "./data"
INDEX_DIR = "./index"
MODEL_NAME = "abhinand/MedEmbed-large-v0.1"

def load_documents():
    documents = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_DIR, filename))
            documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    return chunks

def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    if os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        print("Loading existing FAISS index...")
        vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    else:
        print("Index not found, building new FAISS index...")
        docs = load_documents()
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(INDEX_DIR)
    return vectorstore