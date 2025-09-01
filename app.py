import streamlit as st
import base64
from rag import get_top_contexts
from utils import format_sources
from rag import get_llm_answer, get_embedder, get_index

get_embedder()
get_index()


def show_ref(file_path, page=None, chunk=None):
    if chunk:
        st.markdown(f"#### {file_path} (page {page})")
        st.markdown(chunk)


def make_select_callback(file_name, page, content):
    def _cb():
        st.session_state["selected_pdf"] = file_name
        st.session_state["selected_page"] = page
        st.session_state["selected_chunk"] = content
    return _cb


def show_pdf(file_path, page=None):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    page_anchor = f"#page={page}" if page else ""
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}{page_anchor}" width="700" height="500" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


st.set_page_config(page_title="Medical RAG Chatbot", layout="wide")
st.title("Medical Q&A Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "selected_pdf" not in st.session_state:
    st.session_state["selected_pdf"] = None
    st.session_state["selected_page"] = None

left, right = st.columns([0.6, 0.4], gap="medium")


st.session_state.chat_box = left.container(height=600, border=True)
with left:
    rerank = st.checkbox("Rerank Context")

with st.session_state.chat_box:

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

if query := st.chat_input("What are the side effects of azithromycin?"):
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.session_state.chat_box:
        with st.chat_message("user"):
            st.markdown(query)

    st.session_state["context"] = get_top_contexts(query, rerank=rerank)
    response = get_llm_answer(query, st.session_state["context"])
    # print("LLM response:\n", response)
    # # answer, sources = split_text_and_json(response)
    # print("Parsed answer:", response['answer'])
    # print("Parsed sources:", response['sources'])
    st.session_state["sources"] = response['sources']

    st.session_state["messages"].append(
        {"role": "assistant", "content": response['answer']})
    with st.session_state.chat_box:
        with st.chat_message("assistant"):
            st.markdown(response['answer'])

with right:
    st.markdown("### References")
    if "context" in st.session_state:
        source_map = format_sources(
            st.session_state["context"], st.session_state.get("sources", []))
        # print("Source map:", source_map)
        for i, (file_name, pages) in enumerate(source_map.items()):
            def _page_sort_key(p):
                p = str(p)
                return (0, int(p)) if p.isdigit() else (1, p)
            for page, content in sorted(pages.items(), key=lambda kv: _page_sort_key(kv[0])):
                st.button(
                    f"View {file_name} (page {page})",
                    key=f"view::{i}::{file_name}::{page}",
                    on_click=make_select_callback(file_name, page, content),
                )

    if st.session_state["selected_pdf"] is not None:
        st.markdown("### Preview Context")
        file_path = st.session_state["selected_pdf"]

        chunk = st.session_state["selected_chunk"]
        if isinstance(chunk, list):
            chunk = "\n\n".join(chunk)

        try:
            show_ref(
                file_path, page=st.session_state["selected_page"], chunk=chunk)
        except Exception as e:
            st.error(f"Preview failed: {e}")
