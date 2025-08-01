from langchain_community.document_loaders import UnstructuredMarkdownLoader, TextLoader, UnstructuredWordDocumentLoader
from pathlib import Path

from langgraph.types import Command

from multi_doc_qa_graph.app import State

def load_documents(state: State):
    p = Path(state.get("document_path"))
    state["documents"].clear()
    documents = []
    for f in p.iterdir():
        if f.suffix == '.md':
            md_loader = UnstructuredMarkdownLoader(str(f))
            documents.extend(md_loader.load())
        elif f.suffix == '.txt':
            txt_loader = TextLoader(str(f), encoding="utf-8")
            documents.extend(txt_loader.load())
        elif f.suffix == '.docx':
            docx_loader = UnstructuredWordDocumentLoader(str(f))
            documents.extend(docx_loader.load())
        else:
            print(f"Unsupported file type: {f.suffix}")
    return Command(update={"documents": documents})
