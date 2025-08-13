import json
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from nltk.corpus import gutenberg

from build_vectorstore_gutenberg import SAVE_DIR, load_docs_from_gutenberg, FILES, PARAGRAPHS_PER_FILE, CHUNK_SIZE, \
    CHUNK_OVERLAP

load_dotenv()

# precision: 检索到的结果中，有多少是相关的
# recall: 所有相关文档中，有多少被检索到

# 举个例子（10 篇文档中，相关文档有 4 篇）：
# 你检索了 5 篇，其中 3 篇是相关的
# Precision = 3 / 5 = 0.6（检回的 60% 是对的）
# Recall = 3 / 4 = 0.75（对的文档中有 75% 被找到了）

def evaluate_search(search_fn, queries, top_k=5):
    total_precision, total_recall = 0, 0
    for q in queries:
        retrieved_docs = search_fn(q['query'], top_k=top_k)
        relevant_docs = set(q['relevant_docs']) # recall的分母

        retrieved_docs_name = []
        for retrieved_doc in retrieved_docs:
            if isinstance(retrieved_doc, tuple):
                retrieved_docs_name.append(retrieved_doc[0].metadata['doc_id'])
            else:
                retrieved_docs_name.append(retrieved_doc.metadata['doc_id'])
        retrieved_set = set(retrieved_docs_name) # precision的分母

        hit = retrieved_set & relevant_docs

        precision = len(hit) / len(retrieved_set) if len(retrieved_set) > 0 else 0
        recall = len(hit) / len(relevant_docs) if len(relevant_docs) > 0 else 0

        total_precision += precision
        total_recall += recall
    avg_precision = total_precision / len(queries)
    avg_recall = total_recall / len(queries)
    return avg_precision, avg_recall

vector_store = FAISS.load_local(SAVE_DIR, OpenAIEmbeddings(model="text-embedding-3-small"),
                          allow_dangerous_deserialization=True, normalize_L2=True)

def vector_search(query, top_k):
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={"k": top_k})
    result = retriever.invoke(query)
    print(result)

def vector_search_with_scores(query, top_k):
    ret = vector_store.similarity_search_with_score(query, top_k=top_k)
    return ret

import jieba
def zh_preprocess(text: str):
    return [t for t in jieba.lcut(text) if t.strip()]

docs = load_docs_from_gutenberg(FILES, PARAGRAPHS_PER_FILE)
splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
chunks = splitter.split_documents(docs)
# 补充 chunk 元信息，便于评估
for i, d in enumerate(chunks):
    d.metadata.setdefault("doc_id", d.metadata.get("source", "unknown"))
    d.metadata["chunk_id"] = i


bm25_retriever = BM25Retriever.from_documents(chunks, preprocess_func=zh_preprocess)
def bm25_search(query, top_k):
    bm25_retriever.k=top_k
    ret = bm25_retriever.invoke(query)
    return ret

def hybrid_search(query, top_k):
    vector_result = vector_store.similarity_search_with_score(query, top_k=top_k)
    bm25_result = bm25_search_with_scores(query, top_k=top_k)
    return fuse_linear(vector_result, bm25_result)


from typing import List, Tuple, Dict, Callable
import math

DocumentScore = Tuple["Document", float]

# === 1) 把 (Document, score) 列表转成 {key: (doc, score)} ===
def list_to_map(
    results: List[DocumentScore],
    key_func: Callable = lambda d: d.metadata.get("doc_id")  # 或者用 chunk 级: lambda d: f"{d.metadata['doc_id']}#{d.metadata['chunk_id']}"
) -> Dict[str, DocumentScore]:
    out = {}
    for doc, s in results:
        k = key_func(doc)
        # 如果同 key 出现多次，保留分数更高的那个
        if k not in out or s > out[k][1]:
            out[k] = (doc, s)
    return out

# === 2) 分数归一化（按“每个查询”做 min-max 或 softmax；都映射到[0,1]） ===
def minmax_norm_map(score_map: Dict[str, DocumentScore]) -> Dict[str, float]:
    if not score_map:
        return {}
    vals = [s for _, s in score_map.values()]
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) if hi > lo else 1e-9
    return {k: (s - lo) / rng for k, (_, s) in score_map.items()}

def softmax_norm_map(score_map: Dict[str, DocumentScore], temp: float = 1.0) -> Dict[str, float]:
    if not score_map:
        return {}
    vals = [s for _, s in score_map.values()]
    m = max(vals)
    exps = {k: math.exp((s - m) / temp) for k, (_, s) in score_map.items()}
    Z = sum(exps.values()) + 1e-12
    return {k: v / Z for k, v in exps.items()}

# === 3) 向量分数方向统一（可选）：把余弦内积从[-1,1]映到[0,1]；若是L2距离先转相似度 ===
def to_similarity_scores(
    score_map: Dict[str, DocumentScore],
    mode: str = "ip"  # "ip" | "cos" | "l2"
) -> Dict[str, float]:
    out = {}
    for k, (_, s) in score_map.items():
        if mode in ("ip", "cos"):  # 余弦/内积（越大越相似），先移到 [0,1]（更稳的归一化前处理）
            sim = (s + 1.0) / 2.0  # 如果你的IP已确保在[0,1]，这步可改为 sim = s
        elif mode == "l2":        # L2 距离（越小越相似）
            sim = 1.0 / (1.0 + max(s, 0.0))
        else:
            sim = s
        out[k] = sim
    return out

# === 4) 线性融合：alpha * bm25 + (1-alpha) * vector （两边先各自归一化到[0,1]）===
def fuse_linear(
    vector_results: List[DocumentScore],
    bm25_results: List[DocumentScore],
    key_func: Callable = lambda d: d.metadata.get("doc_id"),
    vec_mode: str = "ip",     # 你的FAISS若是 IndexFlatIP + normalize_L2=True，就用 "ip"
    norm: str = "minmax",     # "minmax" 或 "softmax"
    alpha: float = 0.5,       # BM25权重
) -> List[DocumentScore]:
    v_map_raw = list_to_map(vector_results, key_func)
    b_map_raw = list_to_map(bm25_results, key_func)

    # 向量分数先转成“相似度方向”
    v_sim = to_similarity_scores(v_map_raw, mode=vec_mode)

    # 把两路各自归一化到[0,1]
    if norm == "softmax":
        v_norm = softmax_norm_map({k: (v_map_raw[k][0], v_sim[k]) for k in v_sim})
        b_norm = softmax_norm_map(b_map_raw)
    else:
        v_norm = minmax_norm_map({k: (v_map_raw[k][0], v_sim[k]) for k in v_sim})
        b_norm = minmax_norm_map(b_map_raw)

    # key 并集；缺席的一侧按 0 处理
    keys = set(v_norm) | set(b_norm)
    fused: Dict[str, Tuple[Document, float]] = {}
    for k in keys:
        v = v_norm.get(k, 0.0)
        b = b_norm.get(k, 0.0)
        score = alpha * b + (1 - alpha) * v
        # 选择一个代表性的 Document（优先有的那边）
        doc = (b_map_raw.get(k) or v_map_raw.get(k))[0]
        fused[k] = (doc, score)

    # 排序返回
    return sorted(fused.values(), key=lambda x: x[1], reverse=True)


def bm25_search_with_scores(query, top_k):
    scores = bm25_retriever.vectorizer.get_scores(query)
    top_n = np.argsort(scores)[::-1][:top_k]
    ret = []
    max_score, min_score = scores.max(), scores.min()
    for n in top_n:
        cur_score = scores[n]
        linear_score = (cur_score - min_score)/(max_score - min_score)
        ret.append((bm25_retriever.docs[n], linear_score))
    return ret

if __name__ == '__main__':
    with open("gutenberg_queries.json", 'r', encoding='utf-8') as f:
        queries = json.load(f)
        bm25_precision, bm25_recall = evaluate_search(bm25_search, queries)
        vec_precision, vec_recall = evaluate_search(vector_search_with_scores, queries)
        hybrid_precision, hybrid_recall = evaluate_search(hybrid_search, queries)

        print(f"BM25: Precision={bm25_precision:.2f}, Recall={bm25_recall:.2f}")
        print(f"向量: Precision={vec_precision:.2f}, Recall={vec_recall:.2f}")
        print(f"混合: Precision={hybrid_precision:.2f}, Recall={hybrid_recall:.2f}")

