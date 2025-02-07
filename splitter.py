from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
# from data_loader import load_documents
def split_documents (documents: list[Document]):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 800,
            chunk_overlap = 80,
            length_function = len,
            is_separator_regex= False,
        )
        print("✅Data Splitted")
        return text_splitter.split_documents(documents)
    except Exception as e:
        print(f"Splitter Error: {e}")
        raise
# documents = load_documents()
# chunks = split_documents(documents)
# print(chunks)