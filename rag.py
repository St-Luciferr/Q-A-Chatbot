import os
from dotenv import load_dotenv

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from ingest import get_vectorstore
from langchain_core.prompts import PromptTemplate

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL", "https://models.inference.ai.azure.com")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")


def get_chain():
    vectorstore = get_vectorstore()
    retriever: VectorStoreRetriever = vectorstore.as_retriever(search_kwargs={
                                                               "k": 5})

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        "You are a medical assistant that answers questions using only the provided context.\n\n"
        "Group the answer into separate paragraphs for each distinct PDF document.\n"
        "Each paragraph should contain only information derived from that specific document.\n"
        "At the end of each paragraph, cite the source using this format: (Source: <pdf_name>, pages <page_label in the context>).\n"
        "don't leave any information in the context unaddressed and combine the context from  the same page of the same pdf.\n"
        "If the context does not provide an answer, respond with: 'I cannot answer the query based on the available document context.'\n\n"
        "Context:\n{context}"),
        
        ("human", "{input}")
    ])

    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model=MODEL_NAME,
                     base_url=BASE_URL, temperature=0.0)

    document_prompt = PromptTemplate.from_template(
        "{page_content}\n(Source: {source}, page {page})"
    )
    qa_chain = create_stuff_documents_chain(
        llm, prompt, document_prompt=document_prompt)
    rag_chain = create_retrieval_chain(
        retriever=retriever, combine_docs_chain=qa_chain)
    

    return rag_chain
