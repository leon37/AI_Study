from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.word_document import UnstructuredWordDocumentLoader
from dotenv import load_dotenv
from pathlib import Path

class MultiDocumentLoader:
    def __init__(self):
        load_dotenv()
        self.documents = []
        self.load_documents()

    def load_documents(self):
        p = Path('./test_docs')
        for f in p.iterdir():
            filename = f.name
            if f.suffix == '.md':
                md_loader = UnstructuredMarkdownLoader(f"./test_docs/{filename}")
                self.documents.append(md_loader.load())
            elif f.suffix == '.txt':
                txt_loader = TextLoader(f"./test_docs/{filename}", encoding="utf-8")
                self.documents.append(txt_loader.load())
            elif f.suffix == '.docx':
                docx_loader = UnstructuredWordDocumentLoader(f"./test_docs/{filename}")
                self.documents.append(docx_loader.load())
            else:
                print(f"Unsupported file type: {f.suffix}")
                continue


if __name__ == '__main__':
    # 当前目录
    l = MultiDocumentLoader()
    l.load_documents()
    print(len(l.documents))




