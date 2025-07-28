from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.prompts import ChatPromptTemplate
import faiss
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def init_prompt():
    system_prompt = (
        "你是一个执行问答任务的助手。 "
        "如果你不知道问题的答案，就说你不知道。"
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    return prompt

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

class RAGModel:
    def __init__(self, chunk_size, chunk_lap):
        load_dotenv()
        self.chunk_size = chunk_size
        self.chunk_lap = chunk_lap
        self.chunks = self.get_chunks()
        self.embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')
        self.llm_model = ChatOpenAI(model='gpt-3.5-turbo')
        self.prompt = init_prompt()
    def get_chunks(self):
        text_loader = TextLoader(file_path='test_doc.txt', encoding='utf-8')
        doc = text_loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_lap,
        )
        chunks = splitter.split_documents(doc)
        return chunks

    def get_answer(self, question):
        index = faiss.IndexFlatL2(len(self.embedding_model.embed_query(question)))
        vector_store = FAISS(
            embedding_function=self.embedding_model,
            index=index,
            index_to_docstore_id={},
            docstore=InMemoryDocstore(),
        )

        vector_store.add_documents(documents=self.chunks)

        retriever = vector_store.as_retriever()
        rag_chain = (
                {"context": retriever | format_docs, "input": RunnablePassthrough()}
                | self.prompt
                | self.llm_model
                | StrOutputParser()
        )
        ret = rag_chain.invoke(question)
        print(ret)



if __name__ == '__main__':
    rag = RAGModel(chunk_size=90, chunk_lap=10)
    rag.get_answer("FAISS 是干什么用的？")

