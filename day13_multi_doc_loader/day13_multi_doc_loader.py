from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.word_document import UnstructuredWordDocumentLoader
from dotenv import load_dotenv
from pathlib import Path

from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def init_prompt():
    system_prompt = (
        "你是一个问答助手，以下是相关背景资料，请根据背景资料回答用户问题，"
        "如果找不到答案，就说不知道。\n\n背景资料：\n{context}"
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

class MultiDocumentLoader:
    def __init__(self):
        load_dotenv()
        self.prompt = init_prompt()
        self.documents = []
        self.load_documents()
        self.embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')
        self.llm_model = ChatOpenAI(model='gpt-3.5-turbo')
        index = faiss.IndexFlatL2(len(self.embedding_model.embed_query('hello!')))
        self.vector_store = FAISS(
            embedding_function=self.embedding_model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        self.split_chunk()

    def load_documents(self):
        p = Path('./test_docs')
        for f in p.iterdir():
            filename = f.name
            if f.suffix == '.md':
                md_loader = UnstructuredMarkdownLoader(f"./test_docs/{filename}")
                document = md_loader.load()
                self.documents.extend(document)
            elif f.suffix == '.txt':
                txt_loader = TextLoader(f"./test_docs/{filename}", encoding="utf-8")
                self.documents.extend(txt_loader.load())
            elif f.suffix == '.docx':
                docx_loader = UnstructuredWordDocumentLoader(f"./test_docs/{filename}")
                self.documents.extend(docx_loader.load())
            else:
                print(f"Unsupported file type: {f.suffix}")
                continue

    def split_chunk(self):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
        )

        chunks = splitter.split_documents(self.documents)
        self.vector_store.add_documents(chunks)

    def run(self, question):
        retriever = self.vector_store.as_retriever()
        rag_chain = (
                {"context": retriever | format_docs, "input": RunnablePassthrough()}
                | self.prompt
                | self.llm_model
                | StrOutputParser()
        )
        rsp = rag_chain.invoke(question)
        return rsp




if __name__ == '__main__':
    l = MultiDocumentLoader()
    print("Hello")
    try:
        while True:
            user_input = input("user: ")
            if user_input == "exit":
                print("Goodbye")
                break
            else:
                print(l.run(user_input))
    except KeyboardInterrupt:
        print("GoodBye")




