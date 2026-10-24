# Campus Regulations Assistant
### A Local RAG vs. General LLM Comparison App

A Streamlit application that answers questions about campus regulations using a local document knowledge base, while showing a second answer from the same language model without retrieved documents. The side-by-side comparison helps users see how retrieval-augmented generation (RAG) can ground answers in supplied sources.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Preparing the Knowledge Base](#preparing-the-knowledge-base)
- [Running the Application](#running-the-application)
- [Using the Interface](#using-the-interface)
- [Retrieval and Indexing Details](#retrieval-and-indexing-details)
- [Understanding the Comparison](#understanding-the-comparison)
- [Performance and Caching](#performance-and-caching)
- [Troubleshooting](#troubleshooting)
- [Security and Privacy Notes](#security-and-privacy-notes)
- [Current Limitations](#current-limitations)
- [Potential Improvements](#potential-improvements)
- [License](#license)

---

## Overview

The **Campus Regulations Assistant** is a local-first question-answering demo built with Streamlit, LangChain, Hugging Face sentence embeddings, FAISS, and Ollama.

For each user question, the application presents two answers side by side:

1. **RAG answer** — retrieves relevant chunks from the configured knowledge base and passes them to the language model as context.
2. **General LLM answer** — asks the same configured model the same question without supplying the retrieved campus documents.

The RAG prompt instructs the model to use the retrieved context and explicitly say when the answer is not available in that context. The general-answer prompt has no document context. The application also displays retrieved source names, page numbers where available, text snippets, and elapsed time for each answer.

This design is useful for demonstrations, experimentation, and learning about retrieval-augmented generation. It is not, by itself, a guarantee that an answer is correct, complete, or an official interpretation of campus policy.

## Key Features

- **Side-by-side answer comparison:** compare a document-grounded answer with a general model answer.
- **Local model inference:** communicates with an Ollama server, configured by default at `http://localhost:11434`.
- **Multiple knowledge-base source types:** loads PDF files, Markdown files, and URLs listed in `links.txt`.
- **Semantic retrieval:** embeds document chunks with `sentence-transformers/all-MiniLM-L6-v2` and stores them in FAISS.
- **Persistent vector index:** saves the FAISS index and a JSON manifest so the index can be reused when the source signature is unchanged.
- **Source inspection:** shows retrieved chunk snippets and PDF page numbers when the loader provides page metadata.
- **Streaming responses:** streams generated text into the Streamlit interface.
- **Status checks:** reports whether the Ollama server is reachable and whether the configured model appears to be installed.
- **Index rebuild control:** provides a sidebar button to remove the saved index and trigger a rebuild.
- **Chat-history controls:** keeps the current conversation in Streamlit session state and provides a button to clear it.
- **Performance visibility:** displays timing information for indexing stages and for both answers.
- **Warning visibility:** exposes source-loading and index-persistence warnings in the sidebar.

## How It Works

At a high level, the request flow is:

```text
PDF / Markdown / URLs
        |
        v
Load source documents
        |
        v
Split documents into overlapping chunks
        |
        v
Create embeddings with Hugging Face
        |
        v
Build or reuse a FAISS vector index
        |
        v
User submits a question
        |
        +-------------------------------+
        |                               |
        v                               v
Retrieve top-k chunks              Ask the same Ollama model
from the vector index              without document context
        |                               |
        v                               v
RAG prompt + question              General prompt + question
        |                               |
        v                               v
Stream grounded answer             Stream general answer
        |                               |
        +---------------+---------------+
                        |
                        v
              Display both answers
              and RAG source snippets
```

### RAG answer path

1. The app retrieves up to four chunks using the FAISS-backed retriever.
2. It concatenates their text into a context string, separated by delimiters.
3. It passes the context and question to the RAG prompt.
4. Ollama generates the answer, which is streamed into the UI.
5. The app displays the retrieved source names, available page numbers, and shortened snippets.

The prompt asks the model not to invent facts and to state when the context does not contain the answer. This is a prompt-level instruction, not a formal guarantee against hallucinations.

### General answer path

1. The app sends the same user question to the same configured Ollama model.
2. It uses a separate prompt with no retrieved document context.
3. It streams and displays the response next to the RAG response.

The application runs the RAG response first and the general response second. They are **sequential**, not parallel, so the displayed total waiting time includes both generations.

## Architecture

The supplied script is organized into the following functional sections:

| Component | Responsibility |
|---|---|
| Configuration | Loads environment variables and defines paths and model/index settings. |
| Ollama status check | Requests `/api/tags` from the configured Ollama server and checks the configured model name. |
| Source signature | Creates a SHA-256 fingerprint from recognized source-file metadata and indexing settings. |
| Embedding cache | Caches the Hugging Face embedding model with Streamlit's resource cache. |
| Document ingestion | Loads PDFs, Markdown files, and web pages listed in `links.txt`. |
| Chunking and indexing | Splits text, creates embeddings, builds the FAISS index, and persists it locally. |
| LLM and prompts | Creates a cached `ChatOllama` instance and defines separate RAG and general prompts. |
| Retrieval and generation | Retrieves relevant chunks and streams each answer. |
| Source rendering | Converts document metadata into displayable source records and renders snippets. |
| Streamlit UI | Shows service status, knowledge-base status, chat history, answers, timings, and controls. |

## Technology Stack

| Technology | Role in this project |
|---|---|
| Python | Application language |
| Streamlit | Web interface, session state, caching, and streamed output |
| LangChain Community | PDF/text/web loaders and FAISS vector-store integration |
| LangChain Text Splitters | Recursive character-based chunking |
| LangChain Hugging Face | Embedding integration |
| Sentence Transformers | Semantic text embeddings |
| FAISS | Vector similarity search over embedded chunks |
| LangChain Ollama | Connection to the local Ollama model |
| Ollama | Local model serving and text generation |
| Requests | Ollama availability/model-list check |
| python-dotenv | Loading configuration from a `.env` file |
| Standard library | File operations, hashing, JSON, timing, and path handling |

## Project Structure

The following layout matches the paths and filenames expected by the script. The main Python file can be named, for example, `app.py`.

```text
campus-regulations-assistant/
├── app.py
├── .env                         # Optional local configuration
├── .gitignore                   # Recommended
├── requirements.txt             # Create with the dependencies below
├── Campus Regulations Info/
│   ├── student-handbook.pdf     # Example PDF source
│   ├── examination-rules.md     # Example Markdown source
│   └── links.txt                # Optional: one URL per line
└── faiss_index/                 # Generated and reused by the app
    ├── index.faiss
    ├── index.pkl
    └── manifest.json
```

**Important:** `Campus Regulations Info` and `faiss_index` are resolved relative to the directory containing the Python script. The source folder is created automatically if it does not exist. The index directory is generated when a non-empty knowledge base is successfully indexed.

The `requirements.txt`, `.env`, and `.gitignore` files are recommended project setup files; they are not created by the supplied script itself.

## Requirements

- Python with a version supported by the selected LangChain, Streamlit, Hugging Face, and FAISS packages.
- Ollama installed and running locally, or reachable at a configured Ollama-compatible Ollama endpoint.
- The model named by `OLLAMA_MODEL` downloaded into the Ollama instance.
- Network access on first use if dependencies, the Hugging Face embedding model, or source URLs must be downloaded.
- Read/write permission in the project directory for the saved FAISS index.

The default generation model is `gemma3:4b`. It can be changed using the environment variables described below.

## Installation

### 1. Obtain the project files

Place the supplied Python script in a project directory and name it `app.py` (or use your actual filename consistently in the run command).

### 2. Create a virtual environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Create a `requirements.txt` containing the direct dependencies used by the script:

```text
streamlit
python-dotenv
requests
pypdf
langchain-community
langchain-text-splitters
langchain-huggingface
langchain-ollama
langchain-core
sentence-transformers
faiss-cpu
```

Then install them:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This is a practical starting list, not a lockfile. Exact compatible versions depend on your Python version and platform. For repeatable deployments, test the environment and pin the versions that work together. On some platforms, FAISS installation may require a compatible Python version or a platform-specific package.

### 4. Install and start Ollama

Install Ollama using the instructions for your operating system from the official Ollama website: https://ollama.com/

Start the Ollama application or run:

```bash
ollama serve
```

If the service is already running, you may not need to start it manually.

### 5. Download the configured model

For the default model:

```bash
ollama pull gemma3:4b
```

Use the same model identifier in `OLLAMA_MODEL` as the one available in your Ollama instance.

### 6. Add knowledge-base documents

Create the directory:

```text
Campus Regulations Info/
```

Place the PDFs and Markdown files you want the assistant to search inside it. Optionally, create a `links.txt` file for web sources.

## Configuration

The script calls `load_dotenv()` and reads the following environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_MODEL` | `gemma3:4b` | Ollama model used for both answers |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Base URL for the Ollama API |
| `OLLAMA_NUM_CTX` | `4096` | Context-window setting passed to `ChatOllama` |

Create a `.env` file in the directory from which the app loads environment configuration, for example:

```dotenv
OLLAMA_MODEL=gemma3:4b
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_NUM_CTX=4096
```

These values are optional because the script defines defaults. Restart the Streamlit app after changing configuration so the new settings are picked up.

### Indexing settings

The following values are currently defined directly in the Python source rather than exposed as environment variables:

| Setting | Value | Meaning |
|---|---:|---|
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model for document chunks and retrieval queries |
| `CHUNK_SIZE` | `1000` | Target chunk size used by the recursive splitter |
| `CHUNK_OVERLAP` | `200` | Overlap between adjacent chunks |
| `RETRIEVAL_K` | `4` | Maximum number of chunks retrieved for a question |

The chunk size is a splitter target and should not be interpreted as a strict token count. The splitter's default length measure is character-based.

## Preparing the Knowledge Base

### PDF files

Add `.pdf` files directly to `Campus Regulations Info/`. The app uses `PyPDFLoader` to extract page-based document content. Source page metadata is displayed as a one-based page number when available.

PDFs with scanned pages may require OCR before their content can be searched. The supplied code does not add an OCR stage.

### Markdown files

Add `.md` files directly to the same folder. They are read as UTF-8 text through `TextLoader`.

### Web pages

Create `Campus Regulations Info/links.txt`, with one URL per line:

```text
# Official campus policy pages
https://example.edu/student-handbook
https://example.edu/examination-regulations
```

Blank lines and lines beginning with `#` are ignored. Each listed URL is passed to `WebBaseLoader`. Pages that fail to load produce warnings. Web ingestion depends on network access and on the page content being accessible to the loader; JavaScript-rendered or access-controlled content may not be extracted as expected.

Use trustworthy, current sources for institutional policies. Verify that you are permitted to download and index any content you include.

### Supported source discovery

The source-signature and ingestion logic recognizes files matching `*.pdf`, `*.md`, and the exact filename `links.txt` directly inside the configured folder. It does not recursively scan subdirectories, and other file types are not ingested by the supplied code.

## Running the Application

From the project directory, with the virtual environment activated and Ollama running:

```bash
streamlit run app.py
```

If your Python script has a different filename, replace `app.py` with that filename. Streamlit will print a local URL in the terminal; open it in your browser.

On first run, the app may take longer because it may need to download/load the embedding model, read the source files, split the documents, compute embeddings, and build the FAISS index.

## Using the Interface

### 1. Check Ollama status

The sidebar reports the configured server URL and model name. It indicates whether the server is reachable and whether the configured model appears in the server's model list.

### 2. Check the knowledge base

The sidebar shows source items loaded by the application, performance timings, and any warnings. If no usable documents are found, add supported files or URLs and use **Rebuild Knowledge Base**.

### 3. Ask a question

Enter a question in the chat input. The app first retrieves context and streams the RAG answer, then streams the general LLM answer.

### 4. Compare the answers

Review the two columns:

- **RAG answer:** intended to use the retrieved knowledge-base content.
- **General LLM answer:** generated without that document context.

Use the source expander beneath the RAG answer to inspect the retrieved chunks. Snippets are shortened for display; they are not necessarily the complete source text.

### 5. Rebuild the knowledge base

Use **Rebuild Knowledge Base** after changing source documents or `links.txt`. The button removes the existing index directory, clears the cached ingestion function, and reruns the app so a fresh index can be built.

### 6. Clear chat history

Use **Clear Chat History** to clear messages stored in the current Streamlit session. This does not delete source documents or the persisted FAISS index.

## Retrieval and Indexing Details

### Source signature and index reuse

Before loading the knowledge base, the app creates a SHA-256 signature from:

- The absolute paths of recognized source files.
- Each file's size and modification timestamp in nanoseconds.
- The embedding model name.
- The chunk size and overlap.

If a saved manifest and both FAISS files exist and the manifest signature matches, the app attempts to load the saved index. Otherwise, it loads the sources and rebuilds the index.

**Implementation note:** this signature is based on file metadata, not a hash of each file's contents. A content change that preserves both file size and modification timestamp could go undetected. The signature also does not include every possible indexing dependency or library version.

### Chunking

`RecursiveCharacterTextSplitter` uses a target chunk size of `1000` and overlap of `200`. Overlap helps preserve context that crosses chunk boundaries. The app discards chunks whose text is blank.

### Embeddings and vector search

The embedding model is `sentence-transformers/all-MiniLM-L6-v2`. `FAISS.from_documents()` creates the vector store from the chunks and their embeddings. A retriever is configured to return up to four chunks per question.

The source does not configure a separate reranker, metadata filter, score threshold, or citation-validation stage.

### Persistence

The app saves the local index with LangChain's FAISS persistence method and writes a `manifest.json` containing the source signature, loaded-source summary, chunk count, embedding model, and chunk settings.

The FAISS store uses a pickle file (`index.pkl`) alongside the FAISS index. Loading it with `allow_dangerous_deserialization=True` can execute code if the pickle has been maliciously crafted. Only load index files that you trust and protect the index directory from untrusted modification.

## Understanding the Comparison

This app is a useful demonstration of the difference between **retrieval-grounded generation** and **generation without supplied source context**. It is not a controlled benchmark by itself.

For a fair interpretation:

- Both answers use the same configured model, but the RAG answer receives extra context and the two prompts differ.
- The RAG answer is generated first and the general answer second, so timings can be affected by warm-up, caching, system load, and generation order.
- The displayed time measures each path as implemented; the RAG time includes retrieval and streamed answer generation, while the general time measures the general generation path.
- A retrieved chunk can be irrelevant or incomplete. Retrieval does not guarantee that the model uses the context correctly.
- A general model may already know some relevant facts from training, but its answer is not grounded in the configured local documents.
- There is no built-in automated evaluation of factual accuracy, retrieval precision, answer completeness, or faithfulness.

For a rigorous evaluation, prepare a fixed set of questions with verified answers and source passages. Measure answer quality and retrieval quality separately, and repeat runs under comparable conditions.

## Performance and Caching

The script uses `st.cache_resource` to cache:

- The embedding model.
- The loaded/rebuilt retriever and related ingestion result, keyed by the function arguments including the source signature.
- Ollama LLM clients keyed by model, base URL, and context length.

The app records elapsed time for index reuse checks, document loading, splitting, embedding/index creation, index saving, and each answer path, where applicable.

Performance depends on document count and length, embedding hardware, model size, context length, Ollama configuration, and the machine running the app. The embedding model may be downloaded on first use. The first indexing run is typically more expensive than reusing a saved index.

## Troubleshooting

### “Ollama server is not reachable”

1. Ensure Ollama is installed and running.
2. Check that `OLLAMA_BASE_URL` points to the correct server.
3. Open the configured base URL's Ollama API endpoint `/api/tags` or test it from a terminal.
4. If the server is remote, verify network access and firewall rules.

### “Model is not installed”

Run the pull command for the configured model, for example:

```bash
ollama pull gemma3:4b
```

Then verify that `OLLAMA_MODEL` matches the model name reported by Ollama. Model tags must match the identifiers returned by the Ollama API.

### “No usable documents found”

- Confirm the directory is named exactly `Campus Regulations Info`.
- Ensure it is next to the Python script.
- Put `.pdf` and `.md` files directly in that directory, or create `links.txt`.
- Check the sidebar warnings for individual loading failures.
- Confirm the documents contain extractable text.

### Web pages are missing

- Check that each URL is correct and reachable.
- Confirm the machine has internet access.
- Review the warnings displayed in the sidebar.
- Some sites block automated requests or render content dynamically, which may limit extraction.

### The index does not appear to update

- Click **Rebuild Knowledge Base** after modifying the files.
- Confirm the app can write to the project directory.
- Remember that source detection is based on file path, size, and modification time rather than full content hashes.
- Ensure you are editing files in the folder used by the running script.

### FAISS or package installation errors

- Confirm the active virtual environment is the one used to install and run the app.
- Check that your Python version is supported by the selected package versions.
- Install dependencies from a clean environment and record working versions for reproducibility.

### The answer does not reflect a document

- Inspect the retrieved snippets to see whether the relevant passage was retrieved.
- Check whether the source was successfully ingested.
- Consider whether the question uses terminology that differs from the source.
- The current implementation retrieves only up to four chunks and has no reranking or score threshold.

## Security and Privacy Notes

- **Local inference is not the same as a fully isolated system.** Model generation is sent to the configured Ollama endpoint. If that endpoint is remote, questions and retrieved context may leave the local machine.
- **Web ingestion uses network access.** URLs in `links.txt` are fetched during indexing or re-indexing.
- **Treat indexed content as untrusted input.** A document or web page can contain misleading text or prompt-injection instructions. The prompt asks the model to use context responsibly, but the application does not implement a dedicated prompt-injection defense.
- **Protect the FAISS index.** `allow_dangerous_deserialization=True` enables pickle deserialization when loading the saved index. Do not load index files supplied or modified by untrusted parties.
- **Avoid committing sensitive data.** Keep private documents, local environment files, and generated index artifacts out of public repositories unless intentionally shared.
- **No authentication layer is implemented in the supplied script.** If deploying beyond a trusted local environment, add appropriate access controls and network protections.

A suggested `.gitignore`:

```gitignore
.venv/
__pycache__/
.env
faiss_index/
.streamlit/secrets.toml
```

Add any confidential source documents to `.gitignore` if they should not be committed. Do not ignore files you intentionally want to distribute with the project.

## Current Limitations

The following are limitations of the supplied implementation, not claims about all RAG systems:

1. Only PDF, Markdown, and URLs in `links.txt` are ingested.
2. Source discovery is limited to the top level of the knowledge-base directory.
3. PDF extraction is text-based; OCR is not implemented.
4. Retrieval returns up to four chunks without reranking or a relevance threshold.
5. Source snippets are displayed, but the model is not required to attach formal citations to individual claims.
6. The source signature uses metadata rather than full file-content hashes.
7. RAG and general generation run sequentially.
8. There is no automated evaluation pipeline or test suite shown in the supplied source.
9. Chat history is stored in Streamlit session state; the code does not implement persistent conversation storage.
10. The application has no built-in authentication, role-based access, or multi-user knowledge-base isolation.
11. Index loading enables dangerous pickle deserialization and therefore assumes the saved index is trusted.
12. The UI displays caught exception text, which may reveal implementation details in some deployments.

## Potential Improvements

These are suggested next steps; they are not currently implemented in the supplied code.

- **Stronger source fingerprints:** hash file contents and include relevant index configuration and version metadata.
- **More robust ingestion:** add OCR for scanned PDFs, recursive directory support, file-type validation, and better per-source diagnostics.
- **Improved retrieval:** evaluate chunk sizes, use metadata filters, add reranking, and optionally apply a similarity threshold.
- **Evidence-aware answers:** include source identifiers in the context and ask for claim-level citations; validate that cited passages support the claims.
- **Evaluation suite:** build a fixed test set and track retrieval recall, answer correctness, faithfulness, latency, and failure cases.
- **Safer index handling:** avoid loading pickle artifacts from untrusted sources and review the persistence strategy for the installed LangChain version.
- **Better observability:** log structured errors and timings without exposing sensitive paths or document contents to end users.
- **Configuration management:** expose chunking and retrieval settings through environment variables or a settings file.
- **Testing:** add unit tests for source discovery, source signatures, ingestion failures, index reuse, and source metadata rendering.
- **Deployment hardening:** add authentication, access control, rate limiting, and operational monitoring before exposing the app to other users.
- **Benchmark methodology:** optionally run both answer paths against a repeatable test set and report comparable metrics.

## License

This project is licensed under the **MIT License**. See the [`LICENSE`](LICENSE) file for the complete license text.

The MIT License applies to this project's code, subject to the terms stated in the license. It does not automatically grant redistribution rights for third-party libraries, models, or documents added to the knowledge base. Review the applicable licenses and permissions for those materials separately.

---

## Acknowledgements

This project combines open-source components from the Streamlit, LangChain, Hugging Face Sentence Transformers, FAISS, and Ollama ecosystems. Refer to each project's official documentation for installation details, compatibility notes, and licensing terms.
