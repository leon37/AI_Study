from langchain_community.docstore import InMemoryDocstore

from multi_doc_qa_graph.app.graph_builder import State
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
import faiss

def embedding(state: State):
    load_dotenv()
    embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')
    index = faiss.IndexFlatL2(len(embedding_model.embeddings('hello!')))
    state['vector_store'] = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    state['vector_store'].add_documents(state['chunks'])