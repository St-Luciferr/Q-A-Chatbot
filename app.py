import streamlit as st
import base64
from rag import get_top_contexts
from utils import format_sources
from urllib.parse import quote
from rag import get_llm_answer

def show_pdf(file_path, page=None):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    page_anchor = f"#page={page}" if page else ""
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}{page_anchor}" width="700" height="500" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

st.set_page_config(page_title="Azithromycin RAG Chatbot",layout="wide")
st.title("Medical Q&A Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "selected_pdf" not in st.session_state:
    st.session_state["selected_pdf"] = None
    st.session_state["selected_page"] = None

left, right = st.columns([0.6, 0.4], gap="medium")

with left:
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

if query := st.chat_input("What are the side effects of azithromycin?"):
    st.session_state["messages"].append({"role": "user", "content": query})
    with left:
        with st.chat_message("user"):
            st.markdown(query)

    st.session_state["context"] = get_top_contexts(query)
    answer = get_llm_answer(query, st.session_state["context"])

    st.session_state["messages"].append({"role": "assistant", "content": answer})
    with left:
        with st.chat_message("assistant"):
            st.markdown(answer)

with right:
    st.markdown("### References")
    if "context" in st.session_state:
        source_map = format_sources(st.session_state["context"])
        for name, pages in source_map.items():
            for page in sorted(pages):
                if st.button(f"View {name} (page {page})", key=f"{name}-{page}"):
                    st.session_state["selected_pdf"] = quote(name)
                    st.session_state["selected_page"] = page

    if st.session_state["selected_pdf"]:
        st.markdown("### Preview")
        file_path = f"./data/{st.session_state['selected_pdf']}"
        show_pdf(file_path, page=st.session_state['selected_page'])
