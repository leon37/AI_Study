import numpy as np
import faiss
import openai
from dotenv import load_dotenv
import os

def embedding(query):
    load_dotenv()
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
    with open("sample_corpus.txt", 'r', encoding='utf-8') as f:
        content = f.read()
        content = content.split('\n')
    resp =client.embeddings.create(
        input=content,
        model='text-embedding-3-small',
    )

    data = []
    for item in resp.data:
        data.append(item.embedding)

    data = np.array(data).astype('float32')
    index = faiss.IndexFlatL2(data.shape[1])
    index.add(data)

    query_rsp = client.embeddings.create(input=[query], model='text-embedding-3-small')
    query_vector = np.array([query_rsp.data[0].embedding]).astype('float32')
    _, ind = index.search(query_vector, k=1)
    print(ind)

    return content[ind[0][0]]

if __name__ == '__main__':
    print(embedding("请找出与“新能源替代传统能源”最相关的文本段。"))