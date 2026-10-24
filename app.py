import os
import glob
import json
import time
import hashlib
import shutil

import requests
import streamlit as st
from dotenv import load_dotenv

os.environ.setdefault("USER_AGENT", "rag-chatbot/1.0")

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="RAG vs General LLM",
    page_icon="📚",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDER_PATH = os.path.join(BASE_DIR, "Open Source Info")
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 4


# --------------------------------------------------
# 2. Ollama status check
# --------------------------------------------------

def check_ollama(base_url: str, model: str):
    """Returns (server_running, model_available, installed_models)."""
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=3)
        resp.raise_for_status()
        names = [m.get("name", "") for m in resp.json().get("models", [])]
    except Exception:
        return False, False, []

    wanted = {model}
    if ":" not in model:
        wanted.add(f"{model}:latest")

    return True, any(n in wanted for n in names), names


# --------------------------------------------------
# 3. Track source-file changes
# --------------------------------------------------

def get_source_signature(folder_path: str) -> str:
    """Fingerprint of local source files + indexing settings."""
    os.makedirs(folder_path, exist_ok=True)

    paths = []
    for pattern in ("*.pdf", "*.md", "links.txt"):
        paths.extend(glob.glob(os.path.join(folder_path, pattern)))

    file_info = []
    for path in sorted(set(paths)):
        try:
            stat = os.stat(path)
            file_info.append({
                "path": os.path.abspath(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            })
        except OSError:
            continue

    payload = json.dumps(
        {
            "files": file_info,
            "embedding_model": EMBEDDING_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------
# 4. Cache the embedding model
# --------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


# --------------------------------------------------
# 5. Load documents and build/reuse the FAISS index
# --------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_and_process_all_sources(folder_path: str, source_signature: str):
    """Returns: retriever, loaded_sources, timings, warnings"""
    timings = {}
    warnings = []
    loaded_summary = []

    os.makedirs(folder_path, exist_ok=True)

    manifest_path = os.path.join(INDEX_PATH, "manifest.json")
    faiss_path = os.path.join(INDEX_PATH, "index.faiss")
    pkl_path = os.path.join(INDEX_PATH, "index.pkl")

    # A. Reuse an existing index if sources are unchanged.
    start = time.perf_counter()

    if (
        os.path.exists(manifest_path)
        and os.path.exists(faiss_path)
        and os.path.exists(pkl_path)
    ):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            if manifest.get("source_signature") == source_signature:
                vectorstore = FAISS.load_local(
                    INDEX_PATH,
                    get_embeddings(),
                    allow_dangerous_deserialization=True,
                )
                timings["Index loading"] = time.perf_counter() - start
                retriever = vectorstore.as_retriever(
                    search_kwargs={"k": RETRIEVAL_K}
                )
                return (
                    retriever,
                    manifest.get("loaded_sources", []),
                    timings,
                    warnings,
                )
        except Exception as exc:
            warnings.append(f"Could not reuse saved index: {exc}")

    timings["Index reuse check"] = time.perf_counter() - start

    # B. Load PDF, Markdown, and web documents.
    documents = []
    start = time.perf_counter()

    for pdf_path in glob.glob(os.path.join(folder_path, "*.pdf")):
        try:
            docs = PyPDFLoader(pdf_path).load()
            if docs:
                documents.extend(docs)
                loaded_summary.append(f"PDF: {os.path.basename(pdf_path)}")
        except Exception as exc:
            warnings.append(f"PDF {os.path.basename(pdf_path)}: {exc}")

    for md_path in glob.glob(os.path.join(folder_path, "*.md")):
        try:
            docs = TextLoader(md_path, encoding="utf-8").load()
            if docs:
                documents.extend(docs)
                loaded_summary.append(
                    f"Markdown: {os.path.basename(md_path)}"
                )
        except Exception as exc:
            warnings.append(f"Markdown {os.path.basename(md_path)}: {exc}")

    links_file_path = os.path.join(folder_path, "links.txt")
    if os.path.exists(links_file_path):
        try:
            with open(links_file_path, "r", encoding="utf-8") as f:
                urls = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.strip().startswith("#")
                ]
        except Exception as exc:
            urls = []
            warnings.append(f"Could not read links.txt: {exc}")

        for url in urls:
            try:
                web_docs = WebBaseLoader(url).load()
                if web_docs:
                    documents.extend(web_docs)
                    loaded_summary.append(f"Web: {url}")
            except Exception as exc:
                warnings.append(f"Web {url}: {exc}")

    timings["Document loading"] = time.perf_counter() - start

    if not documents:
        return None, [], timings, warnings

    # C. Split into chunks.
    start = time.perf_counter()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = [
        c for c in splitter.split_documents(documents)
        if c.page_content.strip()
    ]
    timings["Document splitting"] = time.perf_counter() - start

    if not chunks:
        warnings.append("Documents were loaded but contained no text.")
        return None, loaded_summary, timings, warnings

    # D. Embed + build FAISS.
    start = time.perf_counter()
    vectorstore = FAISS.from_documents(chunks, get_embeddings())
    timings["Embedding + index creation"] = time.perf_counter() - start

    # E. Persist the index.
    start = time.perf_counter()
    try:
        os.makedirs(INDEX_PATH, exist_ok=True)
        vectorstore.save_local(INDEX_PATH)
        manifest = {
            "source_signature": source_signature,
            "loaded_sources": loaded_summary,
            "chunk_count": len(chunks),
            "embedding_model": EMBEDDING_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    except Exception as exc:
        warnings.append(f"Could not save the index: {exc}")
    timings["Index saving"] = time.perf_counter() - start

    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    return retriever, loaded_summary, timings, warnings


# --------------------------------------------------
# 6. LLM (Ollama) + the two chains
# --------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_llm(model: str, base_url: str, num_ctx: int):
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=0.2,
        num_ctx=num_ctx,
    )


RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant for question-answering tasks.\n"
    "Use the retrieved context to answer the question.\n"
    "If the answer is not available in the context, clearly state "
    "that it is not available.\n"
    "Do not invent facts or claim that the context contains "
    "information it does not contain.\n"
    "Keep the answer well-structured and concise.\n\n"
    "Context:\n{context}"
)

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    ("human", "{input}"),
])

# Deliberately plain: same model, same tone, but no documents.
GENERAL_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question as well as you can. "
    "Keep the answer well-structured and concise."
)

GENERAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", GENERAL_SYSTEM_PROMPT),
    ("human", "{input}"),
])


def retrieve(retriever, question: str):
    return retriever.invoke(question)


def rag_stream(docs, question: str):
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    llm = get_llm(OLLAMA_MODEL, OLLAMA_BASE_URL, OLLAMA_NUM_CTX)
    chain = RAG_PROMPT | llm | StrOutputParser()
    return chain.stream({"context": context, "input": question})


def general_stream(question: str):
    llm = get_llm(OLLAMA_MODEL, OLLAMA_BASE_URL, OLLAMA_NUM_CTX)
    chain = GENERAL_PROMPT | llm | StrOutputParser()
    return chain.stream({"input": question})


def docs_to_sources(docs):
    """Convert retrieved documents into plain dicts (safe to keep in history)."""
    sources = []
    for doc in docs:
        src = doc.metadata.get("source", "Unknown")
        name = (
            src
            if src.startswith(("http://", "https://"))
            else os.path.basename(src)
        )
        page = doc.metadata.get("page")
        sources.append({
            "name": name,
            "page": page + 1 if isinstance(page, int) else None,
            "snippet": doc.page_content,
        })
    return sources


# --------------------------------------------------
# 7. Rendering helpers
# --------------------------------------------------

def render_sources(sources):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)} retrieved chunks)"):
        for i, s in enumerate(sources, start=1):
            page_str = f" (Page {s['page']})" if s["page"] else ""
            st.markdown(f"**Source {i}:** `{s['name']}`{page_str}")
            text = s["snippet"]
            st.caption(text[:300] + ("..." if len(text) > 300 else ""))


def render_assistant_message(msg):
    col_rag, col_gen = st.columns(2)

    with col_rag:
        st.markdown("### 📚 RAG answer")
        st.caption("Uses your documents as context")
        st.markdown(msg["rag_answer"])
        st.caption(f"Time: {msg['rag_time']:.1f} s")
        render_sources(msg["sources"])

    with col_gen:
        st.markdown("### 🧠 General LLM answer")
        st.caption("Same model, no documents")
        st.markdown(msg["general_answer"])
        st.caption(f"Time: {msg['general_time']:.1f} s")


# --------------------------------------------------
# 8. Streamlit application
# --------------------------------------------------

def main():
    st.title("RAG vs General LLM — Open Source Contribution")
    st.caption(
        f"Both answers come from `{OLLAMA_MODEL}` running locally via "
        "Ollama. The only difference is that the left one is grounded in "
        "your documents."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    server_ok, model_ok, installed = check_ollama(
        OLLAMA_BASE_URL, OLLAMA_MODEL
    )

    with st.sidebar:
        st.header("Ollama Status")
        st.caption(f"Server: `{OLLAMA_BASE_URL}`")
        st.caption(f"Model: `{OLLAMA_MODEL}`")

        if not server_ok:
            st.error("Ollama server is not reachable.")
        elif not model_ok:
            st.error(f"Model `{OLLAMA_MODEL}` is not installed.")
            if installed:
                st.caption("Installed: " + ", ".join(installed))
        else:
            st.success("Ollama is running and the model is available.")

        st.divider()
        st.header("Knowledge Base Status")
        st.caption(f"Folder: `{FOLDER_PATH}`")

        if st.button("Rebuild Knowledge Base", use_container_width=True):
            if os.path.exists(INDEX_PATH):
                shutil.rmtree(INDEX_PATH)
            load_and_process_all_sources.clear()
            st.rerun()

        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    source_signature = get_source_signature(FOLDER_PATH)

    with st.spinner("Loading knowledge base (first run may take longer)..."):
        retriever, loaded_sources, timings, warnings = (
            load_and_process_all_sources(FOLDER_PATH, source_signature)
        )

    with st.sidebar:
        if loaded_sources:
            st.success(f"{len(loaded_sources)} source item(s) loaded")
            for source in loaded_sources:
                st.markdown(f"- {source}")
        else:
            st.warning("No usable documents found.")

        if timings:
            with st.expander("Performance timings"):
                for task, seconds in timings.items():
                    st.write(f"**{task}:** {seconds:.2f} s")

        if warnings:
            with st.expander(f"Warnings ({len(warnings)})", expanded=True):
                for w in warnings:
                    st.warning(w)

    if not retriever:
        st.info(
            f"Add PDF files, Markdown files, or a `links.txt` file "
            f"(one URL per line) inside:\n\n`{FOLDER_PATH}`\n\n"
            f"then click **Rebuild Knowledge Base**."
        )
        return

    if not server_ok:
        st.error(
            f"Cannot reach Ollama at `{OLLAMA_BASE_URL}`. "
            "Start it with `ollama serve` (or open the Ollama app), "
            "then refresh this page."
        )
        return

    if not model_ok:
        st.error(
            f"Model `{OLLAMA_MODEL}` is not installed. "
            f"Run `ollama pull {OLLAMA_MODEL}` and refresh this page."
        )
        return

    # Chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                render_assistant_message(message)

    user_input = st.chat_input(
        "Ask about contributing to your repository..."
    )

    if user_input:
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            try:
                col_rag, col_gen = st.columns(2)

                with col_rag:
                    st.markdown("### 📚 RAG answer")
                    st.caption("Uses your documents as context")
                    rag_box = st.empty()
                    rag_meta = st.empty()
                    rag_sources_box = st.empty()

                with col_gen:
                    st.markdown("### 🧠 General LLM answer")
                    st.caption("Same model, no documents")
                    gen_box = st.empty()
                    gen_meta = st.empty()

                gen_box.info("Waiting for the RAG answer to finish...")

                # 1) RAG answer
                start = time.perf_counter()
                with rag_box.container():
                    with st.spinner("Retrieving context..."):
                        docs = retrieve(retriever, user_input)
                    rag_answer = st.write_stream(
                        rag_stream(docs, user_input)
                    )
                rag_time = time.perf_counter() - start
                rag_meta.caption(f"Time: {rag_time:.1f} s")

                sources = docs_to_sources(docs)
                with rag_sources_box.container():
                    render_sources(sources)

                # 2) General answer
                start = time.perf_counter()
                with gen_box.container():
                    general_answer = st.write_stream(
                        general_stream(user_input)
                    )
                general_time = time.perf_counter() - start
                gen_meta.caption(f"Time: {general_time:.1f} s")

                st.session_state.messages.append({
                    "role": "assistant",
                    "rag_answer": rag_answer,
                    "rag_time": rag_time,
                    "sources": sources,
                    "general_answer": general_answer,
                    "general_time": general_time,
                })
            except Exception as exc:
                st.error(f"An error occurred: {exc}")


if __name__ == "__main__":
    main()