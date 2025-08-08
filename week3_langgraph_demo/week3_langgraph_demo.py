import json
from pathlib import Path

import faiss
from langchain_community.docstore import InMemoryDocstore
from langchain_community.document_loaders import UnstructuredMarkdownLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.types import Command
from typing_extensions import TypedDict, Annotated
from langgraph.graph import add_messages
from dotenv import load_dotenv
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
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
        SystemMessage(content="""
        你是一个问答助手，以下是相关背景资料，请根据背景资料回答用户问题，"
        "如果找不到答案，就说不知道。\n\n背景资料：\n{context}"""),
        HumanMessage(content="{user_input}")
    ]),
    cur_intent='',
    incoming_input='',
)

graph_builder = StateGraph(State)

llm = ChatOpenAI(model='gpt-4o-mini')
embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')


def intent_router(state: State):
    prompt = state['intent_prompt'].format_messages()
    prompt.extend(state['messages'][-3:])
    rsp = llm.invoke(prompt)
    content = rsp.content
    content_json = json.loads(content)
    if content_json['confidence'] > 0.6:
        return Command(update={'cur_intent': content_json['label']})
    return Command(update={'cur_intent': 'unknown'})


def condition_router(state: State):
    return state.get('cur_intent', 'unknown')


def ingest(state: State):
    return Command(update={'messages': HumanMessage(content=state['incoming_input'])})


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
retriever = vector_store.as_retriever(search_kwargs={"k": 1})


def format_doc(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def rag(state: State):
    if state['incoming_input'] == '':
        raise ValueError('user input is empty')

    result = retriever.invoke(state['incoming_input'])
    context = format_doc(result)
    new_messages = state['rag_prompt'].format_messages(context=context, user_input=state['incoming_input'])
    return Command(update={'messages': new_messages})


def output(state: State):
    if len(state['messages']) == 0:
        raise ValueError('no messages')
    rsp = llm.invoke(state['messages'])
    content = rsp.content
    print(content)
    return Command(update={'messages': AIMessage(content=content)})


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
graph_builder.add_node("output_node", output)
graph_builder.add_node("rag_node", rag)

graph_builder.add_edge("ingest_node", "router_node")
graph_builder.add_conditional_edges(
    source="router_node",
    path=condition_router,
    path_map={
        'rag': 'rag_node',
        'qa': 'output_node',
    }
)
graph_builder.add_edge('rag_node', 'output_node')

graph_builder.set_entry_point("ingest_node")
graph_builder.set_finish_point("output_node")

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
