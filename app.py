"""Streamlit RAG chat app using Groq for generation and Chroma for retrieval."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from docx import Document as DocxDocument
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv()

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "chat_history.db"
CHROMA_DIR = DATA_DIR / "chroma"
SUPPORTED_TYPES = ["pdf", "txt", "md", "docx"]


def database() -> sqlite3.Connection:
    DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL,
        role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS documents (
        conversation_id TEXT NOT NULL, file_hash TEXT NOT NULL, name TEXT NOT NULL,
        PRIMARY KEY (conversation_id, file_hash)
    )""")
    conn.commit()
    return conn


def create_conversation() -> str:
    conversation_id = str(uuid.uuid4())
    conn = database()
    conn.execute(
        "INSERT INTO conversations VALUES (?, ?, ?)",
        (conversation_id, "New conversation", datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()
    return conversation_id


def conversation_list() -> list[sqlite3.Row]:
    conn = database()
    rows = conn.execute("SELECT * FROM conversations ORDER BY created_at DESC").fetchall()
    conn.close()
    return rows


def get_messages(conversation_id: str) -> list[sqlite3.Row]:
    conn = database()
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id", (conversation_id,)
    ).fetchall()
    conn.close()
    return rows


def save_message(conversation_id: str, role: str, content: str) -> None:
    conn = database()
    conn.execute(
        "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (conversation_id, role, content, datetime.now(timezone.utc).isoformat()),
    )
    if role == "user":
        count = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ? AND role = 'user'", (conversation_id,)
        ).fetchone()[0]
        if count == 1:
            conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (content[:50], conversation_id))
    conn.commit()
    conn.close()


def saved_document_names(conversation_id: str) -> list[str]:
    conn = database()
    rows = conn.execute("SELECT name FROM documents WHERE conversation_id = ?", (conversation_id,)).fetchall()
    conn.close()
    return [row[0] for row in rows]


def vector_store(conversation_id: str) -> Chroma:
    directory = CHROMA_DIR / conversation_id
    directory.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name="documents",
        persist_directory=str(directory),
    )


def read_upload(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    data = uploaded_file.getvalue()
    if suffix in {".txt", ".md"}:
        return data.decode("utf-8", errors="replace")
    if suffix == ".pdf":
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp:
            temp.write(data)
            temp_path = temp.name
        try:
            return "\n".join(page.extract_text() or "" for page in PdfReader(temp_path).pages)
        finally:
            os.unlink(temp_path)
    if suffix == ".docx":
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as temp:
            temp.write(data)
            temp_path = temp.name
        try:
            return "\n".join(paragraph.text for paragraph in DocxDocument(temp_path).paragraphs)
        finally:
            os.unlink(temp_path)
    raise ValueError(f"Unsupported file: {uploaded_file.name}")


def index_uploads(conversation_id: str, uploaded_files) -> tuple[int, list[str]]:
    if not uploaded_files:
        return 0, []
    conn = database()
    store = vector_store(conversation_id)
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    indexed, skipped = 0, []
    for file in uploaded_files:
        raw = file.getvalue()
        file_hash = hashlib.sha256(raw).hexdigest()
        known = conn.execute(
            "SELECT 1 FROM documents WHERE conversation_id = ? AND file_hash = ?", (conversation_id, file_hash)
        ).fetchone()
        if known:
            skipped.append(file.name)
            continue
        text = read_upload(file)
        if not text.strip():
            skipped.append(f"{file.name} (no extractable text)")
            continue
        chunks = splitter.split_documents([Document(page_content=text, metadata={"source": file.name})])
        store.add_documents(chunks)
        conn.execute("INSERT INTO documents VALUES (?, ?, ?)", (conversation_id, file_hash, file.name))
        indexed += 1
    conn.commit()
    conn.close()
    return indexed, skipped


def answer_question(conversation_id: str, question: str, api_key: str, model: str) -> str:
    store = vector_store(conversation_id)
    docs = store.similarity_search(question, k=5)
    if not docs:
        return "Upload at least one document before asking a question."
    context = "\n\n".join(f"[Source: {doc.metadata.get('source', 'document')}]\n{doc.page_content}" for doc in docs)
    recent = get_messages(conversation_id)[-8:]
    history = "\n".join(f"{item['role'].title()}: {item['content']}" for item in recent)
    prompt = f"""You are a helpful retrieval-augmented document assistant.
Answer only from the document context below. If it does not contain the answer, say so plainly.
Use the conversation history only to understand follow-up questions; never treat it as a factual source.
When useful, name the source file in your answer. Be concise and accurate.

Conversation history:
{history or '(none)'}

Document context:
{context}

Question: {question}
"""
    llm = ChatGroq(groq_api_key=api_key, model_name=model, temperature=0.2)
    return llm.invoke(prompt).content


def document_action(conversation_id: str, action: str, api_key: str, model: str, selected_docs: list[str]) -> str:
    """Run a document-level operation against the selected indexed files."""
    store = vector_store(conversation_id)
    docs = store.get(where={"source": {"$in": selected_docs}}, include=["documents", "metadatas"])
    pages = [
        f"[Source: {metadata.get('source', 'document')}]\n{content}"
        for content, metadata in zip(docs.get("documents") or [], docs.get("metadatas") or [])
        if content
    ]
    if not pages:
        return "I couldn't find text for the selected documents."
    context = "\n\n".join(pages)
    instructions = {
        "Summarise": "Summarise the selected document(s). Cover the main ideas, key facts, and conclusions. Cite the source filename for each point.",
        "Compare": "Compare the selected documents. Explain their shared themes and the important differences, citing source filenames. If only one document is selected, explain that at least two are needed.",
        "Verify": "Check the selected document(s) for factual claims that can be checked against the other selected documents. For each claim, report whether it is supported, contradicted, or not verifiable from these documents, and cite the relevant filenames. Do not use outside knowledge.",
    }
    llm = ChatGroq(groq_api_key=api_key, model_name=model, temperature=0.2)
    prompt = f"""You are a careful document assistant. Use only the supplied document text. Do not invent facts. Be clear when evidence is missing.

Task: {instructions[action]}

Documents:
{context}
"""
    return llm.invoke(prompt).content


st.set_page_config(page_title="Groq Document RAG", page_icon="📚", layout="wide")
st.title("📚 Groq Document RAG Agent")
st.caption("Upload files, ask grounded questions, and return to saved conversations later.")

if "conversation_id" not in st.session_state:
    conversations = conversation_list()
    st.session_state.conversation_id = conversations[0]["id"] if conversations else create_conversation()

with st.sidebar:
    st.header("Workspace")
    if st.button("+ New conversation", use_container_width=True):
        st.session_state.conversation_id = create_conversation()
        st.rerun()
    conversations = conversation_list()
    choices = {row["id"]: row["title"] for row in conversations}
    selected = st.selectbox(
        "Saved chats", list(choices), format_func=lambda key: choices[key],
        index=list(choices).index(st.session_state.conversation_id) if st.session_state.conversation_id in choices else 0,
    )
    if selected != st.session_state.conversation_id:
        st.session_state.conversation_id = selected
        st.rerun()


    st.divider()


    try:
        groq_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        groq_key = os.getenv("GROQ_API_KEY", "")

    api_key = st.text_input(
        "Groq API key",
        value=groq_key,
        type="password"
    )

    model = st.selectbox(
        "Groq model",
        ["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
        index=0
    )


    
toolbar_upload, toolbar_tools = st.columns([4, 1])
with toolbar_upload:
    uploads = st.file_uploader(
        "Add documents",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.conversation_id}"
    )

document_names = saved_document_names(st.session_state.conversation_id)
with toolbar_tools:
    st.write("")
    tools_open = st.button("Document tools", use_container_width=True)

if tools_open:
    st.session_state.show_document_tools = not st.session_state.get("show_document_tools", False)

if st.session_state.get("show_document_tools", False):
    with st.container(border=True):
        st.subheader("Document tools")
        if not document_names:
            st.info("Upload and index documents to use these tools.")
        else:
            selected_docs = st.multiselect(
                "Choose documents",
                document_names,
                default=document_names,
                key=f"document_selection_{st.session_state.conversation_id}",
            )
            action_cols = st.columns(3)
            for column, action in zip(action_cols, ("Summarise", "Compare", "Verify")):
                if column.button(action, key=f"document_action_{action}", use_container_width=True):
                    if not api_key:
                        st.error("Add your Groq API key in the sidebar or .env file.")
                    elif not selected_docs:
                        st.warning("Choose at least one document.")
                    else:
                        progress_label = {
                            "Summarise": "Summarising selected documents...",
                            "Compare": "Comparing selected documents...",
                            "Verify": "Verifying selected documents...",
                        }[action]
                        with st.spinner(progress_label):
                            try:
                                result = document_action(
                                    st.session_state.conversation_id, action, api_key, model, selected_docs
                                )
                            except Exception as exc:
                                result = f"I couldn't complete that request: {exc}"
                        st.markdown(result)

if uploads:
    with st.spinner("Reading and indexing documents..."):
        added, skipped = index_uploads(
            st.session_state.conversation_id,
            uploads
        )

    if added:
        st.success(f"Successfully indexed {added} document(s).")

    if skipped:
        st.info("Already indexed or could not read: " + ", ".join(skipped))

    names = saved_document_names(st.session_state.conversation_id)
    if names:
        st.caption("Indexed: " + ", ".join(names))

for message in get_messages(st.session_state.conversation_id):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask about your documents"):
    if not api_key:
        st.error("Add your Groq API key in the sidebar or .env file.")
    elif not saved_document_names(st.session_state.conversation_id):
        st.warning("Upload and index a document first.")
    else:
        with st.chat_message("user"):
            st.markdown(question)
        save_message(st.session_state.conversation_id, "user", question)
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and thinking..."):
                try:
                    reply = answer_question(st.session_state.conversation_id, question, api_key, model)
                except Exception as exc:
                    reply = f"I couldn't complete that request: {exc}"
            st.markdown(reply)
        save_message(st.session_state.conversation_id, "assistant", reply)
