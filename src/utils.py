
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from pinecone import Pinecone, ServerlessSpec
import time
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# https://smith.langchain.com/hub/rlm/rag-prompt
PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
llm = init_chat_model("llama3-8b-8192", model_provider="groq")

def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return embedding_model

def get_vectorstore():
    index_name = os.environ["PINECONE_INDEX_NAME"]
    embedding_model = get_embedding_model()
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embedding_model)
    return vectorstore

def create_vectorstore_index():
    pc = Pinecone(os.environ["PINECONE_API_KEY"])
    index_name = "pdf-embeddings-v01"
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if os.environ["PINECONE_INDEX_NAME"] not in existing_indexes:
        pc.create_index(
            name=os.environ["PINECONE_INDEX_NAME"],
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
    return True

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = get_vectorstore().similarity_search(state["question"], k=3)
    return {"context": retrieved_docs}


def generate(state: State):
    context_text = "\n---\n".join(doc.page_content for doc in state["context"])
    messages = prompt_template.format(question=state["question"], context=context_text)
    print(messages)
    response = llm.invoke(messages)
    return {"answer": response.content}