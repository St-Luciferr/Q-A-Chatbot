# Setup Instructions

## 1. Install dependencies
```bash
pip install -r requirements.txt
```

## 2. Copy the env-example to .env
```bash
cp .env.example .env
```

## 3. Put the documents into `data/` folder.

## 4. Run document Ingestion:
```bash
python ingest.py
```

## 5. Run the Chatbot:
```
streamlit run app.py
```



## Developer Note: Medical RAG Chatbot with LlamaIndex, Pinecone & Jina Reranker
I designed and implemented a modular Retrieval-Augmented Generation (RAG) chatbot optimized for querying medical documents with high factual accuracy and citation transparency.

The pipeline is built using LlamaIndex as the core document ingestion and indexing framework. I parsed and chunked medical PDFs into overlapping segments using LlamaIndex’s SentenceSplitter, which ensures semantic coherence across chunks. Each chunk was embedded using the **"abhinand/MedEmbed-large-v0.1"** model from Hugging Face — a transformer fine-tuned specifically for the medical domain.

The vector embeddings are stored in Pinecone, a scalable cloud-native vector database. At query time, relevant chunks are retrieved using semantic similarity search. To further improve the precision of the retrieved context, I integrated Jina AI’s reranking service, which reorders the top-k results based on cross-encoder relevance scoring — boosting answer quality and minimizing noise.

The retrieved and reranked context is then passed into an LLM (OpenAI GPT) along with a prompt that instructs the model to:

- Ground its response strictly in the provided content,

- Cite source filenames and page numbers per paragraph,

- Indicate when an answer is not present in the context.

The user interface is built in Streamlit, providing a clean and responsive chat experience. Users can not only ask medical questions in natural language but also preview the source PDFs and navigate to the cited sections.
