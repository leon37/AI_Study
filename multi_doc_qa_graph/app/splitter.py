from langchain_text_splitters import RecursiveCharacterTextSplitter

from multi_doc_qa_graph.app.graph_builder import State


def chunk_splitter(state: State):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    state['chunks'] = splitter.split_documents(state['documents'])