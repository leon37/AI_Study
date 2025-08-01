from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.types import Command

from multi_doc_qa_graph.app import State


def chunk_splitter(state: State):
    if not state.get("documents"):
        raise ValueError("No documents to split.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(state['documents'])
    return Command(update={'chunks': chunks})