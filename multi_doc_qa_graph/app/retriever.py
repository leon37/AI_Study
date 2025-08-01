from dotenv import load_dotenv
from langchain_core.documents import Document
from langgraph.types import Command

from multi_doc_qa_graph.app import State


def retriever_node(state: State):
    question = state.get("question", "")
    if len(question) == 0:
        raise ValueError("no question provided")
    vector_store = state.get("vector_store")
    if vector_store is None:
        raise ValueError("no vector_store provided")

    retriever = vector_store.as_retriever(search_kwargs={"k": 1})
    result = retriever.invoke(question)
    return Command(update={'retrieved_chunks': format_doc(result)})

def format_doc(docs: list[Document])->str:
    return "\n\n".join(doc.page_content for doc in docs)