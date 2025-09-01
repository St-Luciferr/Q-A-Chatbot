import os
import textwrap
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
EMBED_MODEL_NAME = "abhinand/MedEmbed-large-v0.1"

client = OpenAI(api_key=OPENAI_API_KEY,
                base_url=BASE_URL
                )


EMBED_MODEL_NAME = "abhinand/MedEmbed-large-v0.1"


SYSTEM_PROMPT = textwrap.dedent("""
    You are a careful medical assistant. Use ONLY the provided context.
    - Organize the answer into separate paragraphs PER PDF.
    - Each paragraph must include information ONLY from that PDF.
    - At the END of each paragraph, add: (Source: <pdf_name>, pages <p1, p2–p4>).
    - Combine multiple excerpts from the SAME PDF page into one coherent paragraph sentence group.
    - Be concise (≤150 words per page). If no relevant context exists, write exactly:
      "I cannot answer the query based on the available document context."

    Rigor:
    - Do not invent facts, page numbers, or document names.
    - If a claim isn’t supported by the context, omit it.
    - Prefer literal wording from context when critical.
    """).strip()


# -------- singletons as globals --------
_EMBEDDER: HuggingFaceEmbedding | None = None
_PC: Pinecone | None = None
_INDEX = None  # pinecone.Index


def get_embedder() -> HuggingFaceEmbedding:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    return _EMBEDDER


def get_index():
    global _PC, _INDEX
    if _PC is None:
        _PC = Pinecone(api_key=PINECONE_API_KEY)
    if _INDEX is None:
        _INDEX = _PC.Index(INDEX_NAME)
    return _INDEX


def get_top_contexts(query, rerank=True, top_k=5):
    embed_model = get_embedder()
    index = get_index()
    query_emb = embed_model.get_query_embedding(query)
    results = index.query(vector=query_emb, top_k=top_k, include_metadata=True)
    contexts = [match.metadata for match in results.matches]
    if not rerank:
        return contexts
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


json_example = json.dumps(
    {
        "answer": "your answer here",
        "sources": [
            {"file_name": "file1.pdf", "page_labels": [
                "1", "2"]}  # combine pages per PDF
        ],
    },
    indent=2,
)


def build_prompts(query: str, context_str: str) -> str:
    user_prompt = textwrap.dedent(f"""
    QUESTION:
    {query}

    CONTEXT (verbatim excerpts; do not use external knowledge):
    {context_str}

    FORMAT:
    Return EXACTLY two sections in this order:

    ### ANSWER
    (Write the per-PDF paragraphs here, each ending with "(Source: <pdf_name>, pages <...>)")

    ### JSON
    ```json
    {json_example}
    ```
    - In the JSON, aggregate unique pages per file under "page_labels".
    - Ensure the answer text equals the ANSWER section above (same content).
    """).strip()

    return user_prompt


json_schema = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file_name": {"type": "string"},
                    "page_labels": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["file_name", "page_labels"]
            }
        }
    },
    "required": ["answer", "sources"],
    "additionalProperties": False
}


def get_llm_answer(query: str, contexts: list[dict]) -> str:
    context_str = ""
    for doc in contexts:
        source = json.loads(doc["_node_content"])["metadata"]["file_name"]
        page = json.loads(doc["_node_content"])["metadata"]["page_label"]
        text = json.loads(doc["_node_content"])["text"]
        context_str += f"{text}\n(Source: {source}, page {page})\n\n"

    user_prompt = build_prompts(query, context_str)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {user_prompt}"}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "rag_output", "schema": json_schema}
        },
        messages=messages,
        temperature=0
    )
    return json.loads(response.choices[0].message.content.strip())
