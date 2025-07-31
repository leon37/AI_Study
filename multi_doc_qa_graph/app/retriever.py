from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from multi_doc_qa_graph.app.graph_builder import State


def retriever(state: State):
    load_dotenv()
    if 'question' in state:
        embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')
        vector = embedding_model.embed_query(state['question'])
        retriever = state['vector_store'].as_retriever()