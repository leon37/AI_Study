from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langgraph.types import Command
from typing_extensions import TypedDict, Annotated
from langgraph.graph import add_messages
from dotenv import load_dotenv
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain.tools import tool

current_file_path = Path(__file__).resolve()
project_path = current_file_path.parent.parent
test_doc_path = project_path / "test_docs/"

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]
    documents: list[Document]
    chunks: list[Document]
    retrieved_chunks: str
    intent_prompt: ChatPromptTemplate
    rag_prompt: ChatPromptTemplate
    cur_intent: str
    incoming_input: str


initial_state = State(
    messages=[],
    documents=[],
    chunks=[],
    retrieved_chunks="",
    intent_prompt=ChatPromptTemplate.from_messages([
        SystemMessage(content="""
        你是路由器节点。
        标签集合: ["qa", "tool", "rag", "chitchat"]。
        请阅读最近3轮对话，输出JSON：
        {"label": <一个标签>, "confidence": <0-1浮点> }。
        判定标准：
        - 明确请求工具/调用/计算/获取名字 → tool
        - 技术/知识问答 → qa
        - 寒暄/闲聊/非任务 → chitchat
        - 文档/背景资料 -> rag
        若不确定，给出最可能标签但confidence < 0.55。"""
    )]),
    rag_prompt=ChatPromptTemplate.from_messages([
        SystemMessage(content=''),
        HumanMessage(content='')
    ]),
    cur_intent='',
    incoming_input='',
)

graph_builder = StateGraph(State)

llm = ChatOpenAI(model='gpt-4o-mini')

def intent_router(state: State):
    prompt = state['intent_prompt'].format_messages()
    prompt.extend(state['messages'][-3:])
    rsp = llm.invoke(prompt)
    content = rsp.content
    if content['confidence'] > 0.6:
        return Command(update={'cur_intent': rsp.content['label']})
    return Command(update={'cur_intent': 'unknown'})

def condition_router(state: State):
    return state.get('cur_intent', 'unknown')

def ingest(state: State):
    return Command(update={'messages':HumanMessage(content=state['incoming_input']), 'incoming_input':''})

def qa(state: State):
    rsp = llm.invoke(state['messages'])
    print(rsp.content)
    return Command(update={'messages': AIMessage(content=rsp.content)})

@tool
def get_nickname(name):
    """用于获取外号或者名字的工具"""
    record = {
        '胥邈': '小猪',
        '卷卷': '小卷崽'
    }
    return record.get(name, name)

@tool
def calculate():
    """用于计算的工具"""
    return 12345

graph_builder.add_node("ingest_node", ingest)
graph_builder.add_node("router_node", intent_router)

graph_builder.add_edge("ingest_node","router_node")

graph_builder.set_entry_point("ingest_node")
graph_builder.set_finish_point("router_node")

graph = graph_builder.compile()


if __name__ == '__main__':
    while True:
        user_input = input('user: ')
        if user_input.lower().startswith('q'):
            print('GoodBye')
            break
        else:
            initial_state['incoming_input'] = user_input
            graph.invoke(initial_state)


