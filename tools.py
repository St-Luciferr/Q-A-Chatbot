"""
Custom LangChain Tools for the Medical Q&A Chatbot.

This module demonstrates:
- Tools: Custom functions wrapped as LangChain tools
- Tool Calling: LLM decides which tool to use
- Agents: Orchestrating multiple tools for complex tasks
"""

import os
from typing import List, Optional
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor

from ingest import get_vectorstore
from models import MedicalResponse, Source

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL", "https://models.inference.ai.azure.com")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")


# =============================================================================
# Custom Tools
# =============================================================================

@tool
def search_medical_knowledge(query: str) -> str:
    """
    Search the medical knowledge base for information about a topic.
    Use this tool when you need to find medical information about drugs,
    conditions, treatments, or mechanisms of action.

    Args:
        query: The medical question or topic to search for

    Returns:
        Relevant context from the medical documents
    """
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant information found in the medical knowledge base."

    formatted = []
    for doc in docs:
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "unknown")
        formatted.append(f"[{source}, page {page}]\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted)


@tool
def summarize_context(text: str, max_sentences: int = 3) -> str:
    """
    Summarize a long piece of text into key points.
    Use this tool when you have retrieved a lot of information and need
    to condense it into a shorter summary.

    Args:
        text: The text to summarize
        max_sentences: Maximum number of sentences in the summary (default: 3)

    Returns:
        A concise summary of the text
    """
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=MODEL_NAME,
        base_url=BASE_URL,
        temperature=0.0
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"Summarize the following text in {max_sentences} sentences or fewer. "
                   "Focus on the most important medical information."),
        ("human", "{text}")
    ])

    chain = prompt | llm
    response = chain.invoke({"text": text})
    return response.content


@tool
def list_available_sources() -> str:
    """
    List all available source documents in the knowledge base.
    Use this tool when the user asks what documents or sources are available.

    Returns:
        List of available PDF documents
    """
    data_dir = "./data"
    if not os.path.exists(data_dir):
        return "No data directory found."

    files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    if not files:
        return "No PDF documents found in the knowledge base."

    return "Available documents:\n" + "\n".join(f"- {f}" for f in files)


@tool
def get_drug_interactions(drug_name: str) -> str:
    """
    Search for drug interaction information for a specific medication.
    Use this tool when the user asks about drug interactions, contraindications,
    or what medications should not be taken together.

    Args:
        drug_name: The name of the drug to check interactions for

    Returns:
        Information about drug interactions if available
    """
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Search specifically for interaction-related content
    query = f"{drug_name} drug interactions contraindications warnings"
    docs = retriever.invoke(query)

    if not docs:
        return f"No interaction information found for {drug_name}."

    formatted = []
    for doc in docs:
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "unknown")
        # Only include content that mentions interactions or warnings
        content = doc.page_content.lower()
        if any(keyword in content for keyword in ["interaction", "contraindic", "warning", "caution"]):
            formatted.append(f"[{source}, page {page}]\n{doc.page_content}")

    if not formatted:
        return f"No specific interaction warnings found for {drug_name}."

    return "\n\n---\n\n".join(formatted)


# =============================================================================
# Agent Setup
# =============================================================================

def get_tools() -> List:
    """Get all available tools for the agent."""
    return [
        search_medical_knowledge,
        summarize_context,
        list_available_sources,
        get_drug_interactions,
    ]


def get_medical_agent() -> AgentExecutor:
    """
    Create an agent that can use multiple tools to answer medical questions.

    The agent will:
    1. Decide which tool(s) to use based on the user's question
    2. Execute the tools and gather information
    3. Synthesize the information into a coherent response

    Returns:
        An AgentExecutor instance ready to handle queries
    """
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=MODEL_NAME,
        base_url=BASE_URL,
        temperature=0.0
    )

    tools = get_tools()

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a medical assistant with access to a knowledge base about medications.
You have several tools available to help answer questions:

1. search_medical_knowledge: Search the knowledge base for medical information
2. summarize_context: Summarize long texts into key points
3. list_available_sources: Show what documents are available
4. get_drug_interactions: Find drug interaction information

Always cite your sources when providing information.
If you cannot find relevant information, clearly state that.
Be helpful but remind users to consult healthcare professionals for medical advice."""),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    return agent_executor


def query_with_agent(query: str, chat_history: Optional[List] = None) -> str:
    """
    Query the medical assistant using the agent with tools.

    Args:
        query: The user's question
        chat_history: Optional conversation history

    Returns:
        The agent's response
    """
    agent = get_medical_agent()
    result = agent.invoke({
        "input": query,
        "chat_history": chat_history or []
    })
    return result["output"]


# =============================================================================
# Demo / Testing
# =============================================================================

if __name__ == "__main__":
    # Demo: Test the tools individually
    print("=" * 60)
    print("Testing Tools")
    print("=" * 60)

    print("\n1. List Available Sources:")
    print(list_available_sources.invoke({}))

    print("\n2. Search Medical Knowledge:")
    result = search_medical_knowledge.invoke({"query": "azithromycin mechanism of action"})
    print(result[:500] + "..." if len(result) > 500 else result)

    print("\n" + "=" * 60)
    print("Testing Agent")
    print("=" * 60)

    print("\nQuery: What is the mechanism of action of azithromycin?")
    response = query_with_agent("What is the mechanism of action of azithromycin?")
    print(f"\nResponse: {response}")
