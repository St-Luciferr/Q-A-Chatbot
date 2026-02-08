"""
RAG (Retrieval-Augmented Generation) module for the Medical Q&A Chatbot.

This module demonstrates multiple LangChain concepts:
- LCEL (LangChain Expression Language) for chain composition
- Structured Outputs for reliable JSON responses
- Chat Models with conversation memory
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.prompts import PromptTemplate

from ingest import get_vectorstore
from models import MedicalResponse, Source

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL", "https://models.inference.ai.azure.com")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")


def get_llm():
    """Initialize the ChatOpenAI LLM."""
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=MODEL_NAME,
        base_url=BASE_URL,
        temperature=0.0
    )


def get_retriever(k: int = 5) -> VectorStoreRetriever:
    """Get the vector store retriever."""
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": k})


def format_docs(docs) -> str:
    """Format retrieved documents into a string context."""
    formatted = []
    for doc in docs:
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "unknown")
        formatted.append(f"{doc.page_content}\n(Source: {source}, page {page})")
    return "\n\n".join(formatted)


def format_chat_history(chat_history: List[Dict[str, str]]) -> List:
    """Convert chat history dict to LangChain message format."""
    messages = []
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    return messages


# =============================================================================
# LCEL Chain with Structured Outputs
# =============================================================================

def get_structured_chain():
    """
    Create an LCEL chain with structured outputs.

    This demonstrates:
    - LCEL pipe syntax for chain composition
    - Structured outputs using Pydantic models
    - Cleaner, more composable code

    Returns:
        A runnable chain that outputs MedicalResponse objects
    """
    llm = get_llm()
    retriever = get_retriever()

    # Prompt template for structured output
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a medical assistant that answers questions using only the provided context.

Based on the context, provide:
1. A comprehensive answer addressing the question
2. List of sources (file names and page numbers) you used
3. Your confidence level (high/medium/low) based on how well the context answers the question

Context:
{context}"""),
        ("human", "{input}")
    ])

    # LCEL chain with structured output
    # The | operator creates a pipeline: retriever -> format -> prompt -> llm
    chain = (
        {
            "context": retriever | format_docs,
            "input": RunnablePassthrough()
        }
        | prompt
        | llm.with_structured_output(MedicalResponse)
    )

    return chain


# =============================================================================
# LCEL Chain with Conversation Memory
# =============================================================================

def get_conversational_chain():
    """
    Create an LCEL chain with conversation history support.

    This demonstrates:
    - Chat Models with multi-turn conversation
    - MessagesPlaceholder for dynamic history injection
    - LCEL composition with multiple inputs

    Returns:
        A runnable chain that maintains conversation context
    """
    llm = get_llm()
    retriever = get_retriever()

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a medical assistant that answers questions using only the provided context.
Group the answer into separate paragraphs for each distinct PDF document.
Each paragraph should contain only information derived from that specific document.
At the end of each paragraph, cite the source using this format: (Source: <pdf_name>, page #page_label).
If the context does not provide an answer, respond with: 'I cannot answer the query based on the available document context.'

Context:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    # LCEL chain with history
    chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["input"])),
            "chat_history": lambda x: format_chat_history(x.get("chat_history", [])),
            "input": lambda x: x["input"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# =============================================================================
# Original Chain (Backward Compatibility)
# =============================================================================

def get_chain():
    """
    Original chain implementation for backward compatibility.

    This is the existing implementation using create_retrieval_chain.
    Kept for comparison with the new LCEL-based chains.
    """
    vectorstore = get_vectorstore()
    retriever: VectorStoreRetriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a medical assistant that answers questions using only the provided context.\n\n"
         "Group the answer into separate paragraphs for each distinct PDF document.\n"
         "Each paragraph should contain only information derived from that specific document.\n"
         "At the end of each paragraph, cite the source using this format: (Source: <pdf_name>, pages #page_label). "
         "Note: page number is the page_label in the metadata\n"
         "don't leave any information in the context unaddressed and combine the context from the same page of the same pdf.\n"
         "If the context does not provide an answer, respond with: 'I cannot answer the query based on the available document context.'\n\n"
         "Context:\n{context}"),
        ("human", "{input}")
    ])

    llm = get_llm()

    document_prompt = PromptTemplate.from_template(
        "{page_content}\n(Source: {source}, page {page})"
    )
    qa_chain = create_stuff_documents_chain(
        llm, prompt, document_prompt=document_prompt
    )
    rag_chain = create_retrieval_chain(
        retriever=retriever, combine_docs_chain=qa_chain
    )

    return rag_chain


# =============================================================================
# Utility Functions for App Integration
# =============================================================================

def query_with_structured_output(query: str) -> MedicalResponse:
    """
    Query the chatbot and get a structured response.

    Args:
        query: The medical question

    Returns:
        MedicalResponse with answer, sources, and confidence
    """
    chain = get_structured_chain()
    return chain.invoke(query)


def query_with_history(query: str, chat_history: List[Dict[str, str]] = None) -> str:
    """
    Query the chatbot with conversation history.

    Args:
        query: The current question
        chat_history: List of previous messages [{"role": "user/assistant", "content": "..."}]

    Returns:
        The assistant's response string
    """
    chain = get_conversational_chain()
    return chain.invoke({
        "input": query,
        "chat_history": chat_history or []
    })
