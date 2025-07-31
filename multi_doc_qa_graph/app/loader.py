from langchain_community.document_loaders import UnstructuredMarkdownLoader, TextLoader, UnstructuredWordDocumentLoader
from pathlib import Path
from graph_builder import State

def load_documents(state: State):
    p = Path(state["document_path"])
    for f in p.iterdir():
        filename = f.name
        if f.suffix == '.md':
            md_loader = UnstructuredMarkdownLoader(f.name)
            state["documents"].extend(md_loader.load())
        elif f.suffix == '.txt':
            txt_loader = TextLoader(f.name, encoding="utf-8")
            state["documents"].extend(txt_loader.load())
        elif f.suffix == '.docx':
            docx_loader = UnstructuredWordDocumentLoader(f.name)
            state["documents"].extend(docx_loader.load())
        else:
            print(f"Unsupported file type: {f.suffix}")
