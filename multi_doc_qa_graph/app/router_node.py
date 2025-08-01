from multi_doc_qa_graph.app import State


def router_node(state: State):
    return state

def router_condition(state: State):
    if state.get("vector_store"):
        return "qa"
    return "ingest"