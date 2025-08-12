import json
from langchain_community.vectorstores import FAISS
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
    total_result = vector_result + bm25_result



    print(vector_result)
    print(bm25_result)

def bm25_search_with_scores(query, top_k):
    scores = bm25_retriever.vectorizer.get_scores(query)
    top_n = np.argsort(scores)[::-1][:top_k]
    ret = []
    for n in top_n:
        ret.append((bm25_retriever.docs[n], scores[n]))
    return ret

if __name__ == '__main__':
    with open("gutenberg_queries.json", 'r', encoding='utf-8') as f:
        queries = json.load(f)
        # bm25_precision, bm25_recall = evaluate_search(bm25_search, queries)
        # vec_precision, vec_recall = evaluate_search(vector_search_with_scores, queries)
        hybrid_precision, hybrid_recall = evaluate_search(hybrid_search, queries)

        # print(f"BM25: Precision={bm25_precision:.2f}, Recall={bm25_recall:.2f}")
        # print(f"向量: Precision={vec_precision:.2f}, Recall={vec_recall:.2f}")
        # print(f"混合: Precision={hybrid_precision:.2f}, Recall={hybrid_recall:.2f}")

