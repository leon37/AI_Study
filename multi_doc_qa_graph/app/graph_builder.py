
from langgraph.graph import StateGraph
from pygments.lexer import default

from multi_doc_qa_graph.app import State
from multi_doc_qa_graph.app.embedding import embedding
from multi_doc_qa_graph.app.loader import load_documents
from multi_doc_qa_graph.app.splitter import chunk_splitter
from multi_doc_qa_graph.app.retriever import retriever_node
from multi_doc_qa_graph.app.chatbot import chatbot
from multi_doc_qa_graph.app.router_node import router_node, router_condition

graph_builder = StateGraph(State)
graph_builder.add_node("document_loader", load_documents)
graph_builder.add_node("chunk_splitter", chunk_splitter)
graph_builder.add_node("embedding", embedding)
graph_builder.add_node("retriever", retriever_node)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("router_node", router_node)
graph_builder.set_entry_point("router_node")
graph_builder.set_finish_point("chatbot")
graph_builder.add_edge("document_loader", "chunk_splitter")
graph_builder.add_edge("chunk_splitter", "embedding")
graph_builder.add_edge("embedding", "retriever")
graph_builder.add_edge("retriever", "chatbot")
graph_builder.add_conditional_edges(
    source='router_node',
    path=router_condition,
    path_map={
        'qa':'retriever',
        'ingest':'document_loader',
    },
)
graph = graph_builder.compile()

