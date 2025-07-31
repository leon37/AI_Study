from typing import Annotated

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing_extensions import TypedDict
from langgraph.graph import add_messages, StateGraph

from multi_doc_qa_graph.app.embedding import embedding
from multi_doc_qa_graph.app.loader import load_documents
from multi_doc_qa_graph.app.splitter import chunk_splitter


class State(TypedDict):
    question: str
    document_path: str
    documents: list
    chunks: list[Document]
    vector_store: FAISS

initial_state = State(
    document_path="../test_docs/",
    documents=[],
    chunks=[]
)

graph_builder = StateGraph(initial_state)
graph_builder.add_node("document_loader", load_documents)
graph_builder.add_node("chunk_splitter", chunk_splitter)
graph_builder.add_node("embedding", embedding)

