from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from graph_builder import graph
from multi_doc_qa_graph.app import State
from pathlib import Path

current_file_path = Path(__file__).resolve()
project_path = current_file_path.parent.parent
test_doc_path = project_path / "test_docs/"


initial_state = State(
    document_path=test_doc_path,
    documents=[],
    chunks=[]
)

if __name__ == "__main__":
    load_dotenv()
    while True:
        user_input = input('user: ')
        if user_input.lower().startswith('q'):
            break
        initial_state['question'] = user_input
        graph.invoke(initial_state)