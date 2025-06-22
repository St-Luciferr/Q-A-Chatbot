import os
from openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv
from pinecone import Pinecone
import requests
import json

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = "assabet-rag"
BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                base_url=BASE_URL
                )


def get_top_contexts(query, top_k=5):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    embed_model = HuggingFaceEmbedding(
        model_name="abhinand/MedEmbed-large-v0.1")
    query_emb = embed_model.get_query_embedding(query)
    index = pc.Index(INDEX_NAME)
    results = index.query(vector=query_emb, top_k=top_k, include_metadata=True)
    contexts = [match.metadata for match in results.matches]

    headers = {
        "Content-Type": "application/json",
        "Authorization":  f"Bearer {JINA_API_KEY}"
    }
    data = {
        "model": "jina-reranker-v2-base-multilingual",
        "query": query,
        "top_n": 3,
        "documents": [json.loads(c["_node_content"])["text"] for c in contexts],
    }
    try:
        res = requests.post(
            "https://api.jina.ai/v1/rerank", json=data, headers=headers
        )
        res.raise_for_status()
        reranked = res.json()
        reranked = reranked["results"]
        return [contexts[r["index"]] for r in reranked]
    except:
        return contexts


def get_llm_answer(query: str, contexts: list[dict]) -> str:
    context_str = ""
    for doc in contexts:
        source = json.loads(doc["_node_content"])["metadata"]["file_name"] 
        page = json.loads(doc["_node_content"])["metadata"]["page_label"] 
        text = json.loads(doc["_node_content"])["text"]
        context_str += f"{text}\n(Source: {source}, page {page})\n\n"

    system_prompt = f"""You are a medical assistant that answers questions using only the provided context.
        Group the answer into separate paragraphs for each distinct PDF document.
        Each paragraph should contain only information derived from that specific document.
        At the end of each paragraph, cite the source using this format: (Source: <pdf_name>, pages #).
        don't leave any information in the context unaddressed and combine the context from  the same page of the same pdf.
        If the context does not provide an answer, respond with: 'I cannot answer the query based on the available document context.
        Context:\n{context_str}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {query}"}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content.strip()
