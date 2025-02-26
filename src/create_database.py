from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import utils

DOCS_PATH = "data"

def load_documuments(path: str):
    docs = []
    f_names = os.listdir(path)
    for f in f_names:
        if f.endswith(".pdf"):
            loader = PyMuPDFLoader(Path(path) / f)
            docs.extend(loader.load())
    return docs

def split_documuments(docs: list):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""],
    )
    split_docs = text_splitter.split_documents(docs)
    return split_docs

# 1. Load Data
docs = load_documuments(DOCS_PATH)
# 2. Document Split
split_docs = split_documuments(docs)
# 3. Embedding model
embedding_model = utils.get_embedding_model()
# 4. Store Embeddings in VectoreStore
utils.create_vectorstore_index()
vectorstore = utils.get_vectorstore()
vectorstore.add_documents(split_docs)

print(f"Saved {len(split_docs)} chunks to pinecone database")