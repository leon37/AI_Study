from langchain_community.vectorstores import VectorStore
from langchain_core.documents import Document
from langgraph.graph import add_messages
from typing_extensions import TypedDict, Annotated


class State(TypedDict):
    question: str
    messages: Annotated[list, add_messages]
    document_path: str
    documents: list[Document]
    chunks: list[Document]
    vector_store: VectorStore
    retrieved_chunks: str