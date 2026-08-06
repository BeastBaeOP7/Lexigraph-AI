import streamlit as st
import requests
import json
import os
from typing import List, Dict, Any, Optional

# Set page config with custom title, icon, and layout
st.set_page_config(
    page_title="Legal AI Assistant",
    page_icon="⚖️",
    layout="wide"
)

# Base API URL
API_URL = os.environ.get("BACKEND_API_URL", "http://localhost:8000")

# Custom UI Styling
st.markdown("""
<style>
    .chat-header {
        font-family: 'Outfit', 'Inter', sans-serif;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.2rem;
        margin-bottom: 0.2rem;
    }
    .chat-subtitle {
        font-size: 0.95rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .intent-badge {
        display: inline-block;
        background-color: #f1f5f9;
        color: #475569;
        border-radius: 4px;
        padding: 0.2rem 0.5rem;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        margin-right: 0.5rem;
    }
    .source-container {
        border-left: 3px solid #cbd5e1;
        padding-left: 0.8rem;
        margin-bottom: 0.8rem;
        font-size: 0.85rem;
    }
    .source-block {
        background-color: #f8fafc;
        border-left: 3px solid #cbd5e1;
        padding: 0.8rem;
        margin-bottom: 0.6rem;
        border-radius: 4px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to list documents
def get_documents() -> List[Dict[str, Any]]:
    try:
        res = requests.get(f"{API_URL}/documents", timeout=5)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return []

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Fetch active documents
indexed_docs = get_documents()

# --- SIDEBAR: Document Management ---
st.sidebar.markdown("### ⚖️ Legal AI Assistant")
st.sidebar.markdown("Upload documents and manage indexed contracts.")

# Multiple File Uploader
uploaded_files = st.sidebar.file_uploader(
    "Upload Contract PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    upload_clicked = st.sidebar.button("Index Documents", use_container_width=True)
    if upload_clicked:
        for ufile in uploaded_files:
            with st.sidebar.spinner(f"Indexing {ufile.name}..."):
                try:
                    files = {"file": (ufile.name, ufile.getvalue(), "application/pdf")}
                    res = requests.post(f"{API_URL}/upload", files=files, timeout=60)
                    if res.status_code == 200:
                        st.sidebar.success(f"Indexed: {ufile.name}")
                    else:
                        st.sidebar.error(f"Error {ufile.name}: {res.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.sidebar.error(f"Failed to connect for {ufile.name}: {e}")
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(f"#### Indexed Contracts ({len(indexed_docs)})")

if indexed_docs:
    for doc in indexed_docs:
        col1, col2 = st.sidebar.columns([4, 1])
        name = doc["document_name"]
        pages = doc.get("page_count", 1)
        col1.write(f"📄 **{name}**\n*(Pages: {pages})*")
        
        if col2.button("🗑️", key=f"del_{name}"):
            with st.sidebar.spinner(f"Removing {name}..."):
                try:
                    res = requests.delete(f"{API_URL}/documents/{name}", timeout=5)
                    if res.status_code == 200:
                        st.sidebar.success(f"Deleted {name}")
                    else:
                        st.sidebar.error(f"Failed to delete {name}")
                except Exception as e:
                    st.sidebar.error(f"Error: {e}")
            st.rerun()
else:
    st.sidebar.info("No documents indexed yet.")

st.sidebar.markdown("---")
st.sidebar.markdown("#### System Controls")

# Developer Mode Toggle
dev_mode = st.sidebar.toggle("Developer Mode", value=False)

if st.sidebar.button("Clear All Contracts", type="primary", use_container_width=True):
    with st.sidebar.spinner("Clearing storage..."):
        try:
            requests.delete(f"{API_URL}/documents", timeout=10)
            st.sidebar.success("Database reset successful!")
        except Exception as e:
            st.sidebar.error(f"Reset failed: {e}")
    st.rerun()

if st.sidebar.button("Clear Chat History", use_container_width=True):
    st.session_state.messages = []
    st.rerun()


# --- MAIN AREA: Conversational Assistant ---
st.markdown("<div class='chat-header'>Legal AI Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='chat-subtitle'>Grounded Question Answering, Clause Extraction, Summarization, and Contract Comparisons.</div>", unsafe_allow_html=True)

def render_assistant_response(metadata: Dict[str, Any], show_dev: bool) -> None:
    intent = metadata.get("intent", "UNKNOWN")
    conf = metadata.get("confidence_score", 0.0)
    validation = metadata.get("validation_status", "PASSED")
    retrieved = metadata.get("retrieved_chunks", [])
    citations = metadata.get("citations", [])
    meta_dict = metadata.get("metadata", {})
    
    # Latencies
    total_lat = meta_dict.get("total_latency", 0.0)
    ret_lat = meta_dict.get("retrieval_latency", 0.0)
    gen_lat = meta_dict.get("generation_latency", 0.0)
    
    unique_docs = len(set(c.get("document_name") for c in retrieved if c.get("document_name")))
    
    if conf >= 0.90:
        badge = f"🟢 High ({conf*100:.1f}%)"
    elif conf >= 0.60:
        badge = f"🟡 Medium ({conf*100:.1f}%)"
    else:
        badge = f"🔴 Low ({conf*100:.1f}%)"
        
    # Render Developer Info if Developer Mode is enabled
    if show_dev:
        st.markdown(f"**Intent:** `{intent}` &nbsp;|&nbsp; **Confidence:** {badge} &nbsp;|&nbsp; **Validation:** `{validation}`")
        st.markdown(f"**Retrieved Documents:** `{unique_docs}` &nbsp;|&nbsp; **Retrieved Chunks:** `{len(retrieved)}`")
        st.markdown(f"**Total Latency:** `{total_lat:.2f}s` &nbsp;|&nbsp; **Retrieval Latency:** `{ret_lat:.2f}s` &nbsp;|&nbsp; **Generation Latency:** `{gen_lat:.2f}s`")
    
    # Sources Used (Task 2)
    if citations:
        st.write("**Sources Used:**")
        for cit in citations:
            c_doc = cit.get("document_name", "Unknown")
            c_pg = cit.get("page_number", "?")
            c_cl = cit.get("clause_number")
            cl_str = f"Clause {c_cl}" if c_cl else ""
            
            with st.container():
                st.markdown(
                    f"<div class='source-container'>📄 **{c_doc}**<br>Page {c_pg}<br>{cl_str}</div>", 
                    unsafe_allow_html=True
                )
            
    # Expandable Retrieved Context (Task 3: inherit theme, no white background boxes)
    if retrieved:
        with st.expander("▼ Retrieved Context"):
            for idx, chunk in enumerate(retrieved):
                src_doc = chunk.get("document_name", "Unknown")
                src_pg = chunk.get("page_number", "?")
                src_cl = chunk.get("clause_number", "")
                cl_str = f", Clause {src_cl}" if src_cl else ""
                
                st.markdown(f"**Source {idx+1}: {src_doc} | Page {src_pg}{cl_str}**")
                st.code(chunk.get("text", "").strip(), language="text")

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("role") == "assistant" and msg.get("metadata"):
            render_assistant_response(msg["metadata"], dev_mode)

# Chat Input at bottom
user_input = st.chat_input("Type your question, comparison, or extraction request here...")

if user_input:
    # 1. Render User Message
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 2. Process and Render Assistant Message
    with st.chat_message("assistant"):
        with st.spinner("Analyzing contracts..."):
            try:
                if not indexed_docs:
                    st.error("No documents have been indexed yet. Please upload and index a contract PDF first.")
                else:
                    res = requests.post(f"{API_URL}/chat", json={"message": user_input}, timeout=60)
                    if res.status_code == 200:
                        response_body = res.json()
                        
                        answer = response_body.get("answer", "")
                        st.write(answer)
                        
                        render_assistant_response(response_body, dev_mode)
                        
                        # Persist message metadata in session state
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "metadata": response_body
                        })
                    else:
                        st.error(f"Error {res.status_code}: {res.json().get('detail', 'Failed to connect')}")
            except Exception as e:
                st.error(f"Error executing request: {e}")
