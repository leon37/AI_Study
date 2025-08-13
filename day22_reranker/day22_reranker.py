# 测试查询
import json

from dotenv import load_dotenv
from langchain_community.docstore import InMemoryDocstore

from langchain_openai import OpenAIEmbeddings
import faiss
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from day22_dataset import ALL_DOCUMENTS, QUERIES, ALL_TOPICS
from FlagEmbedding import FlagReranker

load_dotenv()
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
dim = len(embeddings.embed_query('hello'))
index = faiss.IndexFlatIP(dim)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    normalize_L2=True,
    index_to_docstore_id={}
)
documents = []
for doc in ALL_DOCUMENTS:
    documents.append(Document(page_content=doc))
vector_store.add_documents(documents)

retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={"k": 10})

print("======BEFORE RERANK======")

before_rerank_record = {
    'A': {},
    'B': {},
    'C': {},
}

after_rerank_record = {
    'A': {},
    'B': {},
    'C': {},
}

for topic in ALL_TOPICS:
    for query in QUERIES[topic]:
        rsp = retriever.invoke(query)
        print(f'TOPIC:{topic} QUERY:{query}')
        all_content = []
        for doc in rsp:
            print(doc.page_content)
            all_content.append(doc.page_content)
        before_rerank_record[topic].update({query: all_content})

reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=False)

for topic, query_info in before_rerank_record.items():
    for query, candidates in query_info.items():
        score_queries = []
        for candidate in candidates:
            score_queries.append([query, candidate])
        scores = reranker.compute_score(score_queries)
        scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        reranked = [c for _, c in scored]
        after_rerank_record.setdefault(topic, {})[query] = reranked

with open("after_rerank_record.json", "w", encoding="utf-8") as f:
    json.dump(after_rerank_record, f, ensure_ascii=False, indent=2)
