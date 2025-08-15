from typing import List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, BaseMessage
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

import tiktoken

load_dotenv()

MODEL = "gpt-4o-mini"
llm = ChatOpenAI(model=MODEL, temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ------- 可配置的“软预算”与策略 -------
SOFT_BUDGET_TOKENS = 8000          # 自定义软预算，而不是依赖 128K 窗口
TRIGGER_RATIO = 0.75               # 触发压缩阈值（占预算比例）
RESERVED_OUTPUT = 1024             # 预留输出 Token
KEEP_RECENT_TURNS = 6              # 永远保留最近 K 条消息（turn=单条 message）

messages: List[BaseMessage] = [SystemMessage(content="你是一个有记忆的助理，回答要准确、简洁。")]

# ---------- token 计数 ----------
def get_encoding_for_model(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # 新模型可能未在 tiktoken 注册，回退到更通用的编码
        try:
            return tiktoken.get_encoding("o200k_base")
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")

def count_tokens_messages(msgs: List[BaseMessage], model: str) -> int:
    enc = get_encoding_for_model(model)
    # 简单、稳妥的估算：把角色+内容拼接后编码
    # （避免使用旧的 per-message 常数，在新模型上可能不准）
    text = []
    for m in msgs:
        role = m.type  # 'system' | 'human' | 'ai'
        text.append(f"{role}:\n{m.content}\n")
    joined = "\n".join(text)
    return len(enc.encode(joined))

# ---------- 消息 <-> 文档 ----------
def messages_to_documents(msgs: List[BaseMessage]) -> List[Document]:
    docs = []
    for idx, m in enumerate(msgs):
        docs.append(Document(
            page_content=m.content,
            metadata={"role": m.type, "idx": idx}
        ))
    return docs

def documents_to_messages(docs: List[Document]) -> List[BaseMessage]:
    out: List[BaseMessage] = []
    for d in docs:
        role = d.metadata.get("role", "human")
        content = d.page_content
        if role in ("human", "user"):
            out.append(HumanMessage(content=content))
        elif role in ("ai", "assistant"):
            out.append(AIMessage(content=content))
        elif role == "system":
            out.append(SystemMessage(content=content))
        else:
            # 未知角色，一律当 human 处理
            out.append(HumanMessage(content=content))
    return out

# ---------- 是否需要压缩 ----------
def should_compress(msgs: List[BaseMessage]) -> bool:
    total = count_tokens_messages(msgs, MODEL) + RESERVED_OUTPUT
    return total > SOFT_BUDGET_TOKENS * TRIGGER_RATIO

# ---------- 执行压缩 ----------
def compress_history(user_input: str):
    global messages
    if not should_compress(messages):
        return

    # 始终保留最近 K 条，压缩更早的
    if len(messages) <= KEEP_RECENT_TURNS:
        return

    older = messages[:-KEEP_RECENT_TURNS]
    recent = messages[-KEEP_RECENT_TURNS:]

    # 把“旧消息”做向量检索 + LLM 抽取压缩
    older_docs = messages_to_documents(older)
    if not older_docs:
        return

    vector = FAISS.from_documents(older_docs, embeddings)
    base_retriever = vector.as_retriever(search_kwargs={"k": 8})  # 你可调 k

    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=compressor
    )

    compressed_docs: List[Document] = compression_retriever.invoke(user_input)

    # 确保按原时间顺序（用 metadata.idx 排序）
    for d in compressed_docs:
        # 某些压缩器可能丢失元数据，做个兜底
        if "idx" not in d.metadata:
            d.metadata["idx"] = 0
    compressed_docs_sorted = sorted(compressed_docs, key=lambda d: d.metadata["idx"])

    # 压缩后的旧消息 + 最近 K 条消息 组成新的 messages
    new_messages = documents_to_messages(compressed_docs_sorted) + recent

    before = count_tokens_messages(messages, MODEL)
    after = count_tokens_messages(new_messages, MODEL)

    messages = new_messages
    print(f"[Compression] tokens: {before} -> {after} | kept recent {KEEP_RECENT_TURNS}")

if __name__ == "__main__":
    try:
        while True:
            user_input = input("user> ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"q", "quit", "exit"}:
                print("Goodbye")
                break

            # 压缩在“写入当前用户输入之前”先尝试触发
            compress_history(user_input)

            # 追加用户消息并调用模型
            messages.append(HumanMessage(content=user_input))
            rsp: AIMessage = llm.invoke(messages)
            print(f"assistant> {rsp.content}")
            messages.append(rsp)

    except KeyboardInterrupt:
        print("\nGoodbye")
