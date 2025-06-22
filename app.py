import streamlit as st
from rag import get_chain
from urllib.parse import quote
import os
import base64

def show_pdf(file_path, page=None):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    page=int(page)+1
    page_anchor = f"#page={page}" if page else ""
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}{page_anchor}" width="700" height="500" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

st.set_page_config(page_title="Azithromycin RAG Chatbot",layout="wide")
st.title("Medicinal Q&A Chatbot")

if "chain" not in st.session_state:
    st.session_state["chain"] = get_chain()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

left, right = st.columns([0.6, 0.4], gap="medium")

with left:
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if query := st.chat_input("What are the side effects of azithromycin?"):
    st.session_state["messages"].append({"role": "user", "content": query})
    with left:
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            result = st.session_state["chain"].invoke({"input": query})
            st.markdown(result["answer"])
        st.session_state["messages"].append({"role": "assistant", "content": result["answer"]})
        st.session_state["last_result"] = result  

with right:
    st.markdown("### Referenced Documents")
    result = st.session_state.get("last_result")
    if result:
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
                if st.button(f"📄 View {label}", key=f"{name}-{page}"):
                    st.session_state["selected_pdf"] = quote(name)
                    st.session_state["selected_page"] = page

        if "selected_pdf" in st.session_state:
            st.markdown("### 📖 Preview")
            file_path = f"./data/{st.session_state['selected_pdf']}"
            show_pdf(file_path, page=st.session_state["selected_page"])
