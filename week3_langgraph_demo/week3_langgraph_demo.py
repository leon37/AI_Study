import json
from pathlib import Path

import faiss
from langchain_community.docstore import InMemoryDocstore
from langchain_community.document_loaders import UnstructuredMarkdownLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict, Annotated
from langgraph.graph import add_messages
from dotenv import load_dotenv
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

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
    clarify_prompt: ChatPromptTemplate
    cur_intent: str
    incoming_input: str
    draft: str
    hitl_result: dict|None


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
        若不确定，给出最可能标签但confidence < 0.55。
        只输出JSON，不要任何额外文字。""")]),

    rag_prompt=ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("""
        你是一个问答助手，以下是相关背景资料，请根据背景资料回答用户问题，
        如果找不到答案，就说不知道。\n\n背景资料：\n{context}"""),
        HumanMessagePromptTemplate.from_template("{user_input}")
    ]),
    clarify_prompt=ChatPromptTemplate.from_messages([
        SystemMessage(content='你现在需要向用户澄清提问或要求用户补充提问关键信息，需要用户强调当前意图标签'),
        HumanMessagePromptTemplate.from_template("{question}")
    ]),

    cur_intent='',
    incoming_input='',
    draft='',
    hitl_result=None,
)

graph_builder = StateGraph(State)

llm = ChatOpenAI(model='gpt-4o-mini')
embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')


def intent_router(state: State):
    prompt = state['intent_prompt'].format_messages()
    prompt += state['messages'][-3:]
    rsp = llm.invoke(prompt)
    content = rsp.content
    try:
        content_json = json.loads(content)
    except Exception:
        return Command(update={'cur_intent': 'unknown'})
    if content_json['confidence'] > 0.6:
        return Command(update={'cur_intent': content_json['label']})
    return Command(update={'cur_intent': 'unknown'})


def condition_router(state: State):
    return state.get('cur_intent', 'unknown')


def ingest(state: State):
    return Command(update={'messages': [HumanMessage(content=state['incoming_input'])]})


def qa(state: State):
    rsp = llm.invoke(state['messages'])
    print(rsp.content)
    return Command(update={'messages': AIMessage(content=rsp.content)})


def load_documents(p):
    documents = []
    for f in p.iterdir():
        if f.suffix == '.md':
            md_loader = UnstructuredMarkdownLoader(str(f))
            documents.extend(md_loader.load())
        elif f.suffix == '.txt':
            txt_loader = TextLoader(str(f), encoding="utf-8")
            documents.extend(txt_loader.load())
        elif f.suffix == '.docx':
            docx_loader = UnstructuredWordDocumentLoader(str(f))
            documents.extend(docx_loader.load())
        else:
            print(f"Unsupported file type: {f.suffix}")
    return documents


def chunk(documents: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(documents)
    return chunks


def embedding() -> VectorStore:
    documents = load_documents(test_doc_path)
    chunks = chunk(documents)
    try:
        sample_embedding = embedding_model.embed_query('help')
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
    vector_store.add_documents(chunks)
    return vector_store


vector_store = embedding()
retriever = vector_store.as_retriever(search_kwargs={"k": 3})


def format_doc(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def rag(state: State):
    if state['incoming_input'] == '':
        raise ValueError('user input is empty')

    if state['retrieved_chunks'] == '':
        result = retriever.invoke(state['incoming_input'])
        context = format_doc(result)
        new_messages = state['rag_prompt'].format_messages(context=context, user_input=state['incoming_input'])
        return Command(update={'messages': new_messages, 'retrieved_chunks': context})
    else:
        new_messages = state['rag_prompt'].format_messages(context=state['retrieved_chunks'], user_input=state['incoming_input'])
        return Command(update={'messages': new_messages})


def output(state: State):
    content = state['draft']
    print(f'Assistant: {content}')
    return Command(update={'messages': AIMessage(content=content)})

def clarify(state: State):
    prompt = state['clarify_prompt'].format_messages(question=state['incoming_input'])
    return Command(update={'messages': prompt})

def final_assemble(state: State):
    last_5_messages = state['messages'][-5:]
    draft = llm.invoke(last_5_messages).content

    decision = interrupt({
        'type': 'hitl_review',
        'payload': {
            'draft': draft,
        }
    })

    return Command(update={'draft': draft, 'hitl_result': decision})

def hitl(state: State):
    r = state.get("hitl_result") or {"action": "approve"}
    draft = state.get("draft", "")

    if r.get("action") == "edit" and r.get("new_text"):
        return {"draft": r["new_text"], "hitl_result": None}
    elif r.get("action") == "reject":
        return {"draft": "（已拒绝发送：请完善需求后重试）", "hitl_result": None}
    else:  # approve
        return {"hitl_result": None}


def htil_condition_router(state: State):
    return state['hitp_flag']

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

tools = [get_nickname, calculate]
llm_with_tools = llm.bind_tools(tools)
tool_map = {tool.name: tool for tool in tools}

def tool_router(state: State):
    rsp = llm_with_tools.invoke(state['messages'][-3:])
    if not rsp.tool_calls:
        return state

    outputs = [rsp]
    for tool_call in rsp.tool_calls:
        tool_name = tool_call['name']
        args = tool_call['args']

        if tool_name not in tool_map:
            return state
        cur_tool = tool_map[tool_name]
        ret = cur_tool.invoke(args)
        tool_msg = ToolMessage(content=json.dumps(ret), name=tool_name, tool_call_id=tool_call['id'])
        outputs.append(tool_msg)
    return Command(update={'messages': outputs})

graph_builder.add_node("ingest_node", ingest)
graph_builder.add_node("router_node", intent_router)
graph_builder.add_node("output_node", output)
graph_builder.add_node("rag_node", rag)
graph_builder.add_node("clarify_node", clarify)
graph_builder.add_node("final_assemble_node", final_assemble)
graph_builder.add_node("hitl_node", hitl)
graph_builder.add_node("tool_router_node", tool_router)

graph_builder.add_edge("ingest_node", "router_node")
graph_builder.add_conditional_edges(
    source="router_node",
    path=condition_router,
    path_map={
        'rag': 'rag_node',
        'qa': 'output_node',
        'unknown': 'clarify_node',
        'tool': 'tool_router_node',
    }
)

graph_builder.add_edge('rag_node', 'final_assemble_node')
graph_builder.add_edge('clarify_node', 'final_assemble_node')
graph_builder.add_edge('tool_router_node', 'final_assemble_node')
graph_builder.add_edge('final_assemble_node', 'hitl_node')
graph_builder.add_edge('hitl_node', 'output_node')

graph_builder.set_entry_point("ingest_node")
graph_builder.set_finish_point("output_node")

memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "1"}}

if __name__ == '__main__':
    while True:
        user_input = input('user: ')
        if user_input.lower().startswith('q'):
            print('GoodBye')
            break
        else:
            initial_state['incoming_input'] = user_input
            for event in graph.stream(initial_state, config):
                if "__interrupt__" in event:
                    payload = list(event["__interrupt__"])[0].value  # 里面有你传的 draft
                    print("\n--- HITL 草稿预览 ---\n", payload.get("draft", ""))

                    # ② CLI 收集人工决策（Web的话在前端收集，然后回传到服务端）
                    sel = input("确认发送？(y=发送 / m=修改 / n=拒绝) > ").strip().lower()
                    if sel.startswith("m"):
                        edited = input("请输入修改后的草稿：\n> ")
                        resume_val = {"action": "edit", "new_text": edited}
                    elif sel.startswith("n"):
                        reason = input("拒绝原因（可空）：\n> ")
                        resume_val = {"action": "reject", "comment": reason}
                    else:
                        resume_val = {"action": "approve"}

                    for ev2 in graph.stream(Command(resume=resume_val), config, stream_mode="updates"):
                        pass
