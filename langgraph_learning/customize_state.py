from typing import Annotated

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]
    name: str
    birthday: str


from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import InjectedToolCallId, tool

from langgraph.types import Command, interrupt, Interrupt

@tool
def human_assistance(
        name: str, birthday: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> str:
    """向人类寻求帮助。"""
    human_response = interrupt({
            "question": "Is this correct?",
            "name": name,
            "birthday": birthday,
        })

        # 🧠 Resume 后，才执行下面的逻辑
    if human_response.get("correct", "").lower().startswith("y"):
        verified_name = name
        verified_birthday = birthday
        response = "Correct"
    else:
        verified_name = human_response.get("name", name)
        verified_birthday = human_response.get("birthday", birthday)
        response = f"Made a correction: {human_response}"

    return Command(update={
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(content=response, tool_call_id=tool_call_id)]
    })

load_dotenv()
tools = [human_assistance]
llm = ChatOpenAI(model='gpt-3.5-turbo')
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    # assert(len(message.tool_calls) <= 1)
    return {"messages": [message]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = InMemorySaver()
config = {"configurable": {"thread_id": "1"}}
graph = graph_builder.compile(checkpointer=memory)

initial_state = State(
    messages=[SystemMessage(content='你是一个帮助用户确认姓名和生日的助手。')]
)

def run_conversation(user_input=None, resume_payload=None):
    if user_input:
        initial_state["messages"].append(HumanMessage(content=user_input))

    iterator = graph.stream(
        resume_payload if resume_payload else initial_state,
        config=config,
    )

    last_messages = None
    for chunk in iterator:
        # 🧨 检测中断
        if isinstance(chunk, dict) and "__interrupt__" in chunk:  # 包含 human_assistance 提供的 payload
            return chunk['__interrupt__'], True  # 直接返回 Command 实例

        if isinstance(chunk, dict) and "messages" in chunk:
            msg = chunk["messages"][-1]
            if isinstance(msg, AIMessage):
                msg.pretty_print(), False
            else:
                print(f"{msg.type.upper()}: {msg.content}")
            last_messages = chunk["messages"], False

    return last_messages, False

if __name__ == "__main__":
    # ✅ 正确执行 LangGraph 的方式
    while True:
        user_input = input("user:")
        if user_input.lower().startswith('q'):
            print('GoodBye')
            break

        result, inter = run_conversation(user_input=user_input)
        if inter:
            print("🔁 Resume 对话: ")
            print(result)
            corrected = {
                "correct": input("Is it correct? (yes/no): "),
                "name": input("Corrected name: "),
                "birthday": input("Corrected birthday: ")
            }
            # 用 resume 参数继续执行
            result = run_conversation(resume_payload=Command(resume=corrected))
        else:
            print(result)
        # 🧼 最后将对话更新回 state
        if isinstance(result, list):  # messages
            initial_state["messages"] = result
