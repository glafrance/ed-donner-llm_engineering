#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import hashlib
import os
import re
import sys
import importlib.util
import subprocess
from urllib.parse import urlparse, unquote

def main():
    def confirm_user(prompt: str):
        """Prompt user for explicit confirmation."""
        print()
        response = input(prompt).strip().lower()
        if response != "continue":
            print("\n  Script execution stopped by user. Exiting safely.\n")
            sys.exit(0)
        print("✅ Confirmation received.\n")

    def ensure_pip():
        """Ensure pip is available and up to date."""
        if importlib.util.find_spec("pip") is None:
            print("pip not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
        else:
            print("pip already installed ✅")

        # Optional: upgrade pip safely
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        except subprocess.CalledProcessError as e:
            print(f" pip upgrade failed ({e}), continuing with existing version.")

    def ensure_packages(packages_map):
        for pip_name, import_name in packages_map.items():
            try:
                importlib.import_module(import_name)
                print(f"✅ {pip_name} already installed (module '{import_name}').")
            except ImportError:
                print(f"⬇️ Installing {pip_name} ...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])
                importlib.import_module(import_name)  # verify
                print(f"✅ Installed {pip_name}.")

    def extract_pdf_urls(md_path: str):
        if not os.path.exists(md_path):
            raise FileNotFoundError(f"README not found at: {md_path}")
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
        # Match .pdf links (including query strings)
        urls = re.findall(r'https?://[^\s)]+\.pdf(?:\?[^\s)]*)?', content, flags=re.IGNORECASE)
        return urls

    def safe_filename_from_url(url: str) -> str:
        parsed = urlparse(url)
        fname = os.path.basename(parsed.path)
        fname = unquote(fname)
        if not fname or not fname.lower().endswith(".pdf"):
            # Fallback: ensure .pdf + short hash for uniqueness
            base = (fname or "file").rstrip(".")
            digest = hashlib.md5(url.encode("utf-8")).hexdigest()[:8]
            if not base.lower().endswith(".pdf"):
                base += ".pdf"
            fname = f"{base[:-4]}_{digest}.pdf"
        # Remove risky characters
        fname = re.sub(r'[^\w.\- ]+', "_", fname)
        return fname

    def download_pdfs(urls, dest_dir):
        saved_paths = []
        for url in urls:
            try:
                filename = safe_filename_from_url(url)
                file_path = os.path.join(dest_dir, filename)

                if os.path.exists(file_path):
                    print(f"⏭️  Skipping (already exists): {file_path}")
                    saved_paths.append(file_path)
                    continue

                print(f"⬇️  Downloading: {url}")
                with requests.get(url, timeout=60, stream=True) as r:
                    r.raise_for_status()
                    with open(file_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 64):
                            if chunk:
                                f.write(chunk)

                print(f"✅ Saved → {file_path}")
                saved_paths.append(file_path)
            except Exception as e:
                print(f"❌ Failed to download {url}: {e}")
        return saved_paths

    def add_metadata(doc, doc_type):
        doc.metadata["doc_type"] = doc_type
        return doc

    def _to_lc_history(history):
        lc = []
        for m in history or []:
            if isinstance(m, dict):
                role = (m.get("role") or "").lower()
                content = m.get("content") or ""
            else:  # tuple/list fallback
                role, content = (m[0] or "").lower(), m[1]
            if role in ("user", "human"):
                lc.append(HumanMessage(content=content))
            elif role in ("assistant", "ai"):
                lc.append(AIMessage(content=content))
        return lc

    def make_chat(rag_chain):
        def chat(question, history):
            if isinstance(question, dict):
                question = question.get("content", "")
            return rag_chain.invoke(question)
        return chat

    # -------------------- Safety confirmations --------------------
    # confirm_user(
    #     "Caution: executing this script downloads PDFs that are believed to be in the public domain,\n"
    #     "but you agree not to violate copyright laws associated with these PDFs.\n"
    #     "Type 'continue' and then press Enter to continue executing this script: "
    # )

    # confirm_user(
    #     "Caution: executing this script may incur costs in using the OpenAI API and other costs.\n"
    #     "To verify you understand that you may incur these costs and to continue executing this script,\n"
    #     "type 'continue' and then press Enter: "
    # )

    # -------------------- Bootstrap --------------------
    ensure_pip()
    print("✅ Environment bootstrap complete.\n")

    print("Checking and installing required packages...\n")
    # pip package -> import module mapping (names differ for some libs)
    PKG_IMPORTS = {
        "chromadb": "chromadb",
        "gradio": "gradio",
        "langchain-chroma": "langchain_chroma",
        "langchain-community": "langchain_community",
        "langchain-core": "langchain_core",
        "langchain-openai": "langchain_openai",
        "langchain-text-splitters": "langchain_text_splitters",
        "numpy": "numpy",
        "pypdf": "pypdf",
        "python-dotenv": "dotenv",
        "requests": "requests",
        "scikit-learn": "sklearn",
        "sentence-transformers": "sentence_transformers",
    }

    ensure_packages(PKG_IMPORTS)

    print("\n✅ All required dependencies are now installed or verified.\n")

    from dotenv import load_dotenv
    import gradio as gr
    from langchain_chroma import Chroma
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_text_splitters import CharacterTextSplitter
    import numpy as np
    import requests

    load_dotenv()

    DATA_DIR = os.getenv("PDF_DATA_DIR", "data_pdfs")
    DB_DIR = os.getenv("CHROMA_DB_DIR", "chroma_db")
    DB_NAME = "vector_db"
    MODEL = "gpt-4o-mini"
    PDF_README_PATH = os.getenv("PDF_README_PATH", "ai-learning-pdfs-readme.md")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DB_DIR, exist_ok=True)

    print(f"📖 Reading: {PDF_README_PATH}")
    pdf_urls = extract_pdf_urls(PDF_README_PATH)
    print(f"🔗 Found {len(pdf_urls)} PDF URLs.")

    pdf_files = download_pdfs(pdf_urls, DATA_DIR)
    documents = []

    for file_path in pdf_files:
        try:
            base = os.path.basename(file_path)
            print(f"📄 Loading: {base}")
            loader = PyPDFLoader(file_path)
            file_docs = loader.load()
            documents.extend([add_metadata(d, base) for d in file_docs])
        except Exception as e:
            print(f"❌ Failed to load {file_path}: {e}")

    if not documents:
        raise ValueError("No documents were loaded. Check PDF paths and ensure 'pypdf' is installed correctly.")
    else:
        print(f"\nTotal documents (pages) loaded: {len(documents)}")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Total number of chunks: {len(chunks)}")

    # Put the chunks of data into a Vector Store that associates a Vector Embedding with each chunk
    # Chroma is a popular open source Vector Database based on SQLLite
    embeddings = OpenAIEmbeddings()

    # If you would rather use the free Vector Embeddings from HuggingFace sentence-transformers
    # Then replace embeddings = OpenAIEmbeddings()
    # with:
    # from langchain.embeddings import HuggingFaceEmbeddings
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Delete if already exists
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()

    # Create vectorstore
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DB_NAME)
    print(f"Vectorstore created with {vectorstore._collection.count()} documents")    

    # Let's investigate the vectors
    collection = vectorstore._collection
    count = collection.count()

    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")

    # create a new Chat with OpenAI
    llm = ChatOpenAI(temperature=0.7, model=MODEL)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    qa_prompt = ChatPromptTemplate.from_template(
        "You are a professional assistant helping persons wishing to learn AI.\n"
        "If the user greets or asks general chit-chat, respond naturally (no context required).\n"
        "Otherwise, use the provided context to answer.\n"
        "If not in the context, you can search the web or say you don't know.\n"
        "But if you cannot come up with an accurate answer just say *I don't know*\n\n"
        "Context:\n{context}\n\nQuestion: {input}"
    )

    rag_chain = (
        {"context": retriever, "input": RunnablePassthrough()}
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    WELCOME = (
        "Hello! I'll help you learn about AI. You can ask me about:\n\n" \
        "- using Python to create AI agents\n" \
        "- how vector databases work\n" \
        "- creating custom chatbots\n" \
        "- creating voice assistants with Vapi"
    )

    # Pre-seed the Chatbot with the welcome as the first assistant message\n",
    chatbot = gr.Chatbot(value=[{"role": "assistant", "content": WELCOME}], type="messages", height=750,);
    gr.ChatInterface(make_chat(rag_chain), chatbot=chatbot, type="messages").launch(inbrowser=True);
    # gr.ChatInterface(make_chat(rag_chain), type="messages").launch(inbrowser=True)

if __name__ == "__main__":
    main()
