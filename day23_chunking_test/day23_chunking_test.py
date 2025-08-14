import json
from pathlib import Path

from langchain_community.docstore import InMemoryDocstore
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, NLTKTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from typing import List
from dotenv import load_dotenv
import faiss
from langchain_community.vectorstores import FAISS

load_dotenv()
DATA_DIR = Path("./day23_mini_benchmark")
embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')
dim = len(embedding_model.embed_query('hello'))


def load_documents():
    docs: List[Document] = []
    # 读取 .md 文档（按文件名排序，保证可复现）
    for f in sorted(DATA_DIR.glob("*.md")):
        # TextLoader 会把整个文件读成一个 Document
        # 如果你要每个段落保留 metadata，可自己按空行分段再 create_documents
        loader = TextLoader(str(f), encoding="utf-8")
        docs.extend(loader.load())

    # 读取 gold 与 sentences
    queries_gold = json.loads((DATA_DIR / "queries_gold.json").read_text(encoding="utf-8"))
    sentences = json.loads((DATA_DIR / "sentences.json").read_text(encoding="utf-8"))
    return docs, queries_gold, sentences


# 1) 固定长度
def gen_fixed_length_chunks(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],  # 优先按语义边界断
    )
    return splitter.split_documents(documents)


# 2) 句子→二次聚合
def gen_sentence_chunks(documents: List[Document]) -> List[Document]:
    # 句子切分
    sent_splitter = NLTKTextSplitter()
    sent_docs = sent_splitter.split_documents(documents)
    # 二次聚合（把若干句合并成接近 512 字符/Token 的块）
    regroup = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    return regroup.split_documents(sent_docs)


# 3) Markdown 结构化：先按头级分，再二次切
def gen_markdown_chunks(documents: List[Document]) -> List[Document]:
    headers = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    out: List[Document] = []
    for doc in documents:
        md_sections = MarkdownHeaderTextSplitter(headers_to_split_on=headers).split_text(
            doc.page_content
        )
        # 给 section 补充来源文件等 metadata
        for sec in md_sections:
            sec.metadata.update(doc.metadata)
        # section 二次裁切，保证长度一致
        small = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50) \
            .split_documents(md_sections)
        out.extend(small)
    return out


docs, queries_gold, sentences = load_documents()
fixed_chunks = gen_fixed_length_chunks(docs)
sentence_chunks = gen_sentence_chunks(docs)
md_chunks = gen_markdown_chunks(docs)

print(f"[Fixed] {len(fixed_chunks)} chunks")
print(f"[Sentence+Regroup] {len(sentence_chunks)} chunks")
print(f"[Markdown struct] {len(md_chunks)} chunks")


def gen_vector_store():
    index = faiss.IndexFlatIP(dim)
    vector_store = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=InMemoryDocstore(),
        normalize_L2=True,
        index_to_docstore_id={}
    )
    return vector_store


fixed_vector_store = gen_vector_store()
sent_vector_store = gen_vector_store()
md_vector_store = gen_vector_store()

fixed_vector_store.add_documents(fixed_chunks)
sent_vector_store.add_documents(sentence_chunks)
md_vector_store.add_documents(md_chunks)

fixed_retriever = fixed_vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 5})
sent_retriever = sent_vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 5})
md_retriever = md_vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 5})

import os
import csv
from collections import defaultdict
from typing import List, Dict, Tuple, Set

def filename_of(doc: Document) -> str:
    """从 metadata['source'] 提取文件名"""
    src = doc.metadata.get("source", "")
    return os.path.basename(src) if src else ""

def build_gold_text_index(sentences: Dict[str, List[str]], queries_gold: List[Dict]) -> Dict[str, List[Tuple[str, int, str]]]:
    """
    为每个 query 构造 gold 三元组列表：(doc_name, sent_id, sent_text)
    sentences: {doc_name: [sent0, sent1, ...]}
    """
    q_gold = {}
    for q in queries_gold:
        qid = q["qid"]
        triples = []
        for block in q["gold"]:
            doc = block["doc"]
            for sid in block["sent_ids"]:
                print(f'doc: {doc}, sid: {sid}')
                if sid >= len(sentences):
                    sid = -1
                sent_text = sentences[doc][sid]
                triples.append((doc, sid, sent_text))
        q_gold[qid] = triples
    return q_gold

def eval_retriever(
    retriever,
    strategy_name: str,
    queries_gold: List[Dict],
    sentences: Dict[str, List[str]],
    top_k: int = 5
) -> List[Dict]:
    """
    对单个 retriever 进行评估，返回每条 query 的度量结果字典列表
    """
    q_gold = build_gold_text_index(sentences, queries_gold)
    rows = []
    for q in queries_gold:
        qid = q["qid"]
        query = q["query"]
        # 运行检索（LangChain 的 retriever 默认返回 List[Document]）
        hits: List[Document] = retriever.invoke(query)  # 或 retriever.get_relevant_documents(query)
        hits = hits[:top_k]

        # 统计 gold 集合
        gold_triples = q_gold[qid]  # [(doc, sid, text), ...]
        gold_set: Set[Tuple[str, int]] = {(doc, sid) for doc, sid, _ in gold_triples}
        gold_count = len(gold_set)

        # 命中的 gold 句子（按“子串 + 来源文件匹配”）
        covered: Set[Tuple[str, int]] = set()
        # 为了减少重复计算，先按文档分组 gold 句
        gold_by_doc: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        for doc, sid, txt in gold_triples:
            gold_by_doc[doc].append((sid, txt))

        # 逐 chunk 检查是否覆盖任一 gold 句
        hit_chunk_count = 0
        for ch in hits:
            ch_file = filename_of(ch)
            ch_text = ch.page_content
            # 看该 chunk 是否覆盖至少一条 gold 句
            covered_this_chunk = False
            for doc, items in gold_by_doc.items():
                if ch_file != doc:
                    continue
                for sid, sent_text in items:
                    # 句子很短时可适当放宽（比如忽略大小写或去空白），这里先用严格子串
                    if sent_text and sent_text in ch_text:
                        covered.add((doc, sid))
                        covered_this_chunk = True
            if covered_this_chunk:
                hit_chunk_count += 1

        # 计算指标
        recall = len(covered) / gold_count if gold_count else 0.0
        precision = hit_chunk_count / len(hits) if hits else 0.0

        rows.append({
            "strategy": strategy_name,
            "qid": qid,
            "query": query,
            "gold_count": gold_count,
            "retrieved": len(hits),
            "hit_chunks": hit_chunk_count,
            "covered_gold_sentences": len(covered),
            "recall": round(recall, 4),
            "precision": round(precision, 4),
        })
    return rows

if __name__ == "__main__":
    # === 跑三种策略 ===
    all_rows = []
    all_rows += eval_retriever(fixed_retriever, "fixed", queries_gold, sentences, top_k=5)
    all_rows += eval_retriever(sent_retriever, "sentence+regroup", queries_gold, sentences, top_k=5)
    all_rows += eval_retriever(md_retriever, "markdown_struct", queries_gold, sentences, top_k=5)

    # 汇总打印（micro 平均）
    from statistics import mean
    def micro_avg(rows, strategy):
        sub = [r for r in rows if r["strategy"] == strategy]
        return {
            "strategy": strategy,
            "avg_recall": round(mean(r["recall"] for r in sub), 4),
            "avg_precision": round(mean(r["precision"] for r in sub), 4),
        }

    summary = [micro_avg(all_rows, s) for s in ["fixed", "sentence+regroup", "markdown_struct"]]
    print("== Averages ==")
    for s in summary:
        print(s)

    # 保存到 CSV
    out_path = Path("./chunk_eval_results.csv")
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved per-query results to {out_path.resolve()}")