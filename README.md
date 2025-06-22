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


## Developer Note: 
I developed a context-aware Retrieval-Augmented Generation (RAG) chatbot tailored for interacting with medical documents. The system is designed with a strong emphasis on accuracy, traceability, and factual reliability.

To begin, I used LangChain to orchestrate the pipeline and FAISS as the vector store for its speed, cost-efficiency, and ease of local reproducibility. Medical documents were parsed into overlapping text chunks to preserve context across boundaries. These chunks were then embedded using the **"abhinand/MedEmbed-large-v0.1"** model from Hugging Face, which is specifically fine-tuned for medical-domain semantic understanding.

For each user query, the system retrieves the top-k semantically relevant chunks from FAISS using similarity search. These are injected into a carefully crafted prompt and passed to OpenAI’s GPT model with a low temperature setting to minimize hallucinations and encourage consistency. The prompt directs the model to generate responses strictly grounded in the retrieved content and to include per-paragraph source citations, followed by a summarized reference list.

The frontend was built with Streamlit, providing an intuitive and responsive interface for querying medical knowledge in natural language.
