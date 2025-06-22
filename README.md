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



