import os, sqlite3, hashlib, pickle, random
from dotenv import load_dotenv
from pydantic import PrivateAttr

load_dotenv()

import nltk
from nltk.corpus import gutenberg
nltk.download("gutenberg", quiet=True)

from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
import faiss

# -------------------------
# 可调参数
# -------------------------
FILES = [
    "carroll-alice.txt",
    "melville-moby_dick.txt",
    "austen-emma.txt",
]  # 先用 3 本，够做 Demo；需要更多可加
PARAGRAPHS_PER_FILE = 300      # 每本最多取多少段（越小越快）
CHUNK_SIZE = 1000              # 建议 800~1200
CHUNK_OVERLAP = 50             # 先小点
SAVE_DIR = "vs_gutenberg_demo" # 向量库保存目录
SEED = 42

# -------------------------
# 简易段落切分 & 采样
# -------------------------
def load_docs_from_gutenberg(files, paras_per_file, seed=SEED):
    random.seed(seed)
    docs = []
    for fid in files:
        raw = gutenberg.raw(fid)
        # 以空行切段；也可更严格用 regex
        paras = [p.strip() for p in raw.split("\n\n") if p.strip()]
        if len(paras) > paras_per_file:
            paras = random.sample(paras, paras_per_file)
        # 合并成一篇长文也行；但更贴近 RAG 的是保留段落，再做 chunk
        text = "\n\n".join(paras)
        docs.append(Document(page_content=text, metadata={"doc_id": fid}))
    return docs

# -------------------------
# 嵌入缓存（SQLite + pickle）
# -------------------------
class SQLiteEmbeddingCache:
    def __init__(self, path="embedding_cache.sqlite"):
        self.conn = sqlite3.connect(path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS cache (h TEXT PRIMARY KEY, v BLOB)")
        self.conn.commit()

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get_many(self, texts):
        hits, misses, order = [], [], []
        for t in texts:
            h = self._hash(t)
            cur = self.conn.execute("SELECT v FROM cache WHERE h=?", (h,))
            row = cur.fetchone()
            if row:
                hits.append(pickle.loads(row[0]))
                order.append(("hit", h))
            else:
                misses.append((t, h))
                order.append(("miss", h))
        return hits, misses, order

    def set_many(self, pairs):
        # pairs: list[(hash, vector)]
        self.conn.executemany("INSERT OR REPLACE INTO cache(h, v) VALUES(?, ?)",
                              [(h, pickle.dumps(vec)) for h, vec in pairs])
        self.conn.commit()

# 包装 OpenAIEmbeddings，透明使用缓存
class CachedOpenAIEmbeddings(OpenAIEmbeddings):
    _cache: SQLiteEmbeddingCache = PrivateAttr()
    def __init__(self, cache: SQLiteEmbeddingCache, **kwargs):
        super().__init__(**kwargs)
        self._cache = cache

    def embed_documents(self, texts):
        cached_vecs, misses, order = self._cache.get_many(texts)
        new_vecs = []
        if misses:
            batch_texts = [t for t, _ in misses]
            new_vecs = super().embed_documents(batch_texts)
            self._cache.set_many([(h, v) for (_, h), v in zip(misses, new_vecs)])
        # 还原原始顺序
        it_hit, it_new = iter(cached_vecs), iter(new_vecs)
        out = []
        for tag, _ in order:
            out.append(next(it_hit) if tag == "hit" else next(it_new))
        return out

    def embed_query(self, text):
        # 可选：也给 query 上缓存
        cached, misses, order = self._cache.get_many([text])
        if cached:
            return cached[0]
        vec = super().embed_query(text)
        self._cache.set_many([(self._cache._hash(text), vec)])
        return vec

# -------------------------
# 构建 & 保存向量库（余弦相似度）
# -------------------------
def build_and_save_vectorstore(docs, save_dir):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    # 补充 chunk 元信息，便于评估
    for i, d in enumerate(chunks):
        d.metadata.setdefault("doc_id", d.metadata.get("source", "unknown"))
        d.metadata["chunk_id"] = i

    cache = SQLiteEmbeddingCache("embedding_cache.sqlite")
    embeddings = CachedOpenAIEmbeddings(cache=cache, model="text-embedding-3-small")

    # 余弦相似度 = IndexFlatIP + 向量单位化
    # 维度稳妥做法：先 embed 一个 token 求 dim（避免不同模型维度差异）
    dim = len(embeddings.embed_query("dim_probe"))
    index = faiss.IndexFlatIP(dim)

    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        normalize_L2=True,   # 关键：单位化后 IP 等价于 cosine
    )
    vectorstore.add_documents(chunks)
    os.makedirs(save_dir, exist_ok=True)
    vectorstore.save_local(save_dir)
    print(f"✅ 向量库已保存到: {save_dir}")

# -------------------------
# 入口
# -------------------------
if __name__ == "__main__":
    docs = load_docs_from_gutenberg(FILES, PARAGRAPHS_PER_FILE)
    build_and_save_vectorstore(docs, SAVE_DIR)

    # 下次直接加载（示例）：
    vs = FAISS.load_local(SAVE_DIR, OpenAIEmbeddings(model="text-embedding-3-small"),
                          allow_dangerous_deserialization=True, normalize_L2=True)
    print(vs.similarity_search("White Rabbit", k=3))
