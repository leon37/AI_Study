from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

# https://langchain-ai.github.io/langgraph/tutorials/get-started/3-add-memory/#2-compile-the-graph

load_dotenv()
memory = InMemorySaver()
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)
llm = ChatOpenAI(model='gpt-3.5-turbo')

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)


graph = graph_builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "1"}}
# def stream_graph_updates(user_input: str):
#     events = graph.stream(
#         {"messages": [{"role": "user", "content": user_input}]},
#         config,
#         stream_mode="value"
#     )
#     for event in events:
#         event["messages"][-1].pretty_print()

user_input = "你好我叫李涛。"

# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()

user_input = "记得我的名字吗？"

# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()

snapshot = graph.get_state(config=config)
print(snapshot)