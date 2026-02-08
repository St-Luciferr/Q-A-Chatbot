"""
Medical Q&A Chatbot - Streamlit Application

This application demonstrates multiple LangChain concepts:
- Chat Models with conversation memory
- LCEL (LangChain Expression Language)
- Structured Outputs
- Tools & Agents

Users can choose between different modes:
1. Standard RAG Chain
2. LCEL Chain with Structured Outputs
3. Agent with Tools
"""

import streamlit as st
import base64
import os

from rag import get_chain, get_structured_chain, get_conversational_chain
from tools import get_medical_agent

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Medical Q&A Chatbot",
    page_icon="💊",
    layout="wide"
)

st.title("Medical Q&A Chatbot")
st.caption("Powered by LangChain - Demonstrating Chat Models, LCEL, Structured Outputs, and Tools/Agents")


# =============================================================================
# Sidebar - Mode Selection
# =============================================================================

with st.sidebar:
    st.header("Configuration")

    mode = st.selectbox(
        "Select Chain Mode",
        options=[
            "Standard RAG",
            "LCEL + Structured Output",
            "Conversational (with memory)",
            "Agent with Tools"
        ],
        help="Choose the LangChain implementation to use"
    )

    st.divider()

    st.markdown("""
    ### LangChain Concepts Demo

    **Standard RAG**: Original retrieval chain

    **LCEL + Structured**: Uses pipe syntax and Pydantic models

    **Conversational**: Maintains chat history context

    **Agent with Tools**: LLM decides which tools to use
    """)

    st.divider()

    if st.button("Clear Chat History"):
        st.session_state["messages"] = []
        st.session_state.pop("last_result", None)
        st.rerun()


# =============================================================================
# Helper Functions
# =============================================================================

def show_pdf(file_path: str, page: int = None):
    """Display a PDF in an iframe."""
    try:
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
        page = int(page) + 1 if page else 1
        page_anchor = f"#page={page}"
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}{page_anchor}" width="100%" height="500" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"PDF file not found: {file_path}")


def process_standard_rag(query: str) -> dict:
    """Process query using standard RAG chain."""
    if "chain" not in st.session_state:
        st.session_state["chain"] = get_chain()

    result = st.session_state["chain"].invoke({"input": query})
    return {
        "answer": result["answer"],
        "context": result.get("context", []),
        "mode": "Standard RAG"
    }


def process_structured(query: str) -> dict:
    """Process query using LCEL chain with structured outputs."""
    chain = get_structured_chain()
    result = chain.invoke(query)
    return {
        "answer": result.answer,
        "sources": result.sources,
        "confidence": result.confidence,
        "mode": "LCEL + Structured Output"
    }


def process_conversational(query: str, history: list) -> dict:
    """Process query with conversation history."""
    chain = get_conversational_chain()
    result = chain.invoke({
        "input": query,
        "chat_history": history
    })
    return {
        "answer": result,
        "mode": "Conversational"
    }


def process_agent(query: str) -> dict:
    """Process query using agent with tools."""
    agent = get_medical_agent()
    result = agent.invoke({
        "input": query,
        "chat_history": []
    })
    return {
        "answer": result["output"],
        "mode": "Agent with Tools"
    }


# =============================================================================
# Session State Initialization
# =============================================================================

if "messages" not in st.session_state:
    st.session_state["messages"] = []


# =============================================================================
# Main Layout
# =============================================================================

left, right = st.columns([0.6, 0.4], gap="medium")

# Chat Interface
with left:
    # Display chat history
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "metadata" in message:
                meta = message["metadata"]
                if "confidence" in meta:
                    confidence_color = {"high": "green", "medium": "orange", "low": "red"}
                    st.caption(f"Confidence: :{confidence_color.get(meta['confidence'], 'gray')}[{meta['confidence']}]")
                if "mode" in meta:
                    st.caption(f"Mode: {meta['mode']}")

    # Chat input
    if query := st.chat_input("Ask a medical question (e.g., What are the side effects of azithromycin?)"):
        # Add user message
        st.session_state["messages"].append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Process based on selected mode
        with st.chat_message("assistant"):
            with st.spinner(f"Processing with {mode}..."):
                try:
                    if mode == "Standard RAG":
                        result = process_standard_rag(query)
                    elif mode == "LCEL + Structured Output":
                        result = process_structured(query)
                    elif mode == "Conversational (with memory)":
                        history = [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state["messages"][:-1]  # Exclude current query
                        ]
                        result = process_conversational(query, history)
                    else:  # Agent with Tools
                        result = process_agent(query)

                    # Display the answer
                    st.markdown(result["answer"])

                    # Show metadata
                    metadata = {"mode": result.get("mode", mode)}
                    if "confidence" in result:
                        confidence_color = {"high": "green", "medium": "orange", "low": "red"}
                        st.caption(f"Confidence: :{confidence_color.get(result['confidence'], 'gray')}[{result['confidence']}]")
                        metadata["confidence"] = result["confidence"]

                    st.caption(f"Mode: {result.get('mode', mode)}")

                    # Save to session state
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": result["answer"],
                        "metadata": metadata
                    })
                    st.session_state["last_result"] = result

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": error_msg,
                        "metadata": {"mode": mode, "error": True}
                    })

# Source Documents Panel
with right:
    st.markdown("### Referenced Documents")

    result = st.session_state.get("last_result")
    if result:
        # Handle different result formats
        if "context" in result and result["context"]:
            # Standard RAG format
            source_documents = {}
            for doc in result["context"]:
                name = os.path.basename(doc.metadata.get("source", "Unknown.pdf"))
                page = str(doc.metadata.get("page", "unknown"))
                if name not in source_documents:
                    source_documents[name] = set()
                source_documents[name].add(page)

            for name, pages in source_documents.items():
                for page in sorted(pages):
                    label = f"{name} (page {page})"
                    if st.button(f"View {label}", key=f"{name}-{page}"):
                        st.session_state["selected_pdf"] = name
                        st.session_state["selected_page"] = page

        elif "sources" in result and result["sources"]:
            # Structured output format
            st.markdown("**Sources used:**")
            for source in result["sources"]:
                st.markdown(f"- **{source.file_name}**: pages {', '.join(source.page_labels)}")

        # Show PDF preview if selected
        if "selected_pdf" in st.session_state:
            st.markdown("### Preview")
            file_path = f"./data/{st.session_state['selected_pdf']}"
            show_pdf(file_path, page=st.session_state.get("selected_page", 0))
    else:
        st.info("Ask a question to see referenced documents here.")


# =============================================================================
# Footer
# =============================================================================

st.divider()
st.caption("""
**Disclaimer**: This chatbot is for educational purposes only. Always consult healthcare professionals for medical advice.

**LangChain Concepts Demonstrated**:
- Chat Models (GPT-4o-mini via Azure)
- LCEL (LangChain Expression Language)
- Structured Outputs (Pydantic models)
- Tools & Agents
""")
