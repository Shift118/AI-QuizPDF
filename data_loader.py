from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    TextLoader,
)
from langchain_community.vectorstores.utils import filter_complex_metadata
import os

DATA_PATH = "Data\\Documents"


def load_pdf(file_path):
    try:
        loader = PyPDFLoader(file_path)
        return loader.load()
    except Exception as e:
        print(f"PDF Error ({file_path}): {str(e)}")
        return []


def load_docx(file_path):
    try:
        loader = UnstructuredWordDocumentLoader(file_path, mode="elements")
        docs = loader.load()
        for doc in docs:
            doc.metadata.pop("emphasized_text_contents", None)
            doc.metadata.pop("emphasized_text_tags", None)
            doc.metadata.pop("languages", None)
        filtered_docs = filter_complex_metadata(docs)
        return filtered_docs
    except Exception as e:
        print(f"DOCX Error ({file_path}): {str(e)}")
        return []


def get_loader(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    return {
        ".pdf": load_pdf,
        ".docx": load_docx,
        ".ppt": lambda f: UnstructuredPowerPointLoader(f).load(),
        ".pptx": lambda f: UnstructuredPowerPointLoader(f).load(),
        ".txt": lambda f: TextLoader(f).load(),
    }.get(ext)


def load_documents():
    documents = []
    for root, _, files in os.walk(DATA_PATH):
        for file in files:
            path = os.path.join(root, file)
            if loader := get_loader(path):
                try:
                    docs = loader(path)
                    valid = [d for d in docs if d.page_content.strip()]
                    documents.extend(valid)
                    print(f"✅ Loaded {len(valid)} pages from {file}")
                except Exception as e:
                    print(f"❌ Failed {file}: {str(e)}")
    return documents


# Usage in your project:
# from data_loader import load_documents
# documents = load_documents()
# print(documents)
