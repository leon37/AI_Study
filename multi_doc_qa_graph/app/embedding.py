from langchain_community.docstore import InMemoryDocstore
from langgraph.types import Command

from multi_doc_qa_graph.app import State
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
import faiss

def embedding(state: State):
    chunks = state.get('chunks', [])
    if not chunks:
        raise ValueError('Chunks not set.')

    embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')
    try:
        sample_embedding = embedding_model.embed_query('hellp')
    except Exception as e:
        raise RuntimeError(f"Failed to embed query: {e}")
    embedding_dim = len(sample_embedding)

    index = faiss.IndexFlatL2(embedding_dim)
    vector_store = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    vector_store.add_documents(state['chunks'])
    print(f"Added {len(chunks)} documents to vector store.")
    return Command(update={"vector_store": vector_store})