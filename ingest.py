import os
from pinecone import Pinecone,  ServerlessSpec
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]

INDEX_NAME = "assabet-rag"
DATA_DIR = "./data"


def ingest_documents():
    docs = SimpleDirectoryReader(DATA_DIR).load_data()

    # Embed using hugging face medembed model fine tuned for medical documents
    embed_model = HuggingFaceEmbedding(
        model_name="abhinand/MedEmbed-large-v0.1")
    text_splitter = SentenceSplitter.from_defaults(
        chunk_size=500,
        chunk_overlap=100
    )

    pc = Pinecone(api_key=PINECONE_API_KEY, spec=ServerlessSpec(
        region="us-east-1",
        scale_to_zero=True,))
    if not pc.has_index(INDEX_NAME):
        pc.create_index(INDEX_NAME, dimension=1024, metric="cosine", spec=ServerlessSpec(
            region="us-east-1",
            cloud="aws"
        ))
    pinecone_index = pc.Index(INDEX_NAME)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[text_splitter])
    print(f"Ingested {len(docs)} documents into Pinecone.")
    return index


if __name__ == "__main__":
    ingest_documents()
