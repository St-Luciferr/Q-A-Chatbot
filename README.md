# Q&A Chatbot

An interactive chatbot that answers questions based on your custom documents using retrieval-augmented generation (RAG) and LLMs.

---

##  Project Setup

### 1. Clone the repository
```bash
git clone https://github.com/St-Luciferr/Q-A-Chatbot.git
cd Q-A-Chatbot
```
### 2. Set up a virtual environment
Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```
Linux/macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
### 3. Install the required dependencies
```bash
pip install -r requirements.txt
```
### 4. Configure environment variables
Copy the example environment config to .env:
```bash
cp .env.example .env
``` 
You need to modify values in .env with your values.

## 📁 Document Setup (Required for chatbot to work)
This project uses local documents as the knowledge base.
### 1. Create a folder named `data/`:
```bash
mkdir data
```
### 2. Place your pdf documents inside the `data/` folder.
⚠️ The `data/` folder is excluded from repository.

## Run Document Ingestion
This will parse the documents and create a searchable vector index:
```bash
python ingest.py
```
This step must be repeated if you add or change documents in the data/ folder.
##  Launch the Chatbot
Start the chatbot interface using Streamlit:
```bash
streamlit run app.py
```
Open the link that appears in your terminal (http://localhost:8501) in a web browser.

## Usage Demo:
The Streamlit-based chat interface presents the conversation history on the left and dynamically displays the reference documents related to the current response on the right, including the ability to preview cited PDF documents directly within the interface.

![demo](./assets/running_demo.JPG)

## Developer Note: Medical RAG Chatbot with LlamaIndex, Pinecone & Jina Reranker
I designed and implemented a modular Retrieval-Augmented Generation (RAG) chatbot optimized for querying medical documents with high factual accuracy and citation transparency.

The pipeline is built using LlamaIndex as the core document ingestion and indexing framework. I parsed and chunked medical PDFs into overlapping segments using LlamaIndex’s SentenceSplitter, which ensures semantic coherence across chunks. Each chunk was embedded using the **"abhinand/MedEmbed-large-v0.1"** model from Hugging Face — a transformer fine-tuned specifically for the medical domain.

The vector embeddings are stored in Pinecone, a scalable cloud-native vector database. At query time, relevant chunks are retrieved using semantic similarity search. To further improve the precision of the retrieved context, I integrated Jina AI’s reranking service, which reorders the top-k results based on cross-encoder relevance scoring — boosting answer quality and minimizing noise.

The retrieved and reranked context is then passed into an LLM (OpenAI GPT) along with a prompt that instructs the model to:

- Ground its response strictly in the provided content,

- Cite source filenames and page numbers per paragraph,

- Indicate when an answer is not present in the context.

The user interface is built in Streamlit, providing a clean and responsive chat experience. Users can not only ask medical questions in natural language but also preview the source PDFs and navigate to the cited sections.
