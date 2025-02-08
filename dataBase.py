from embeddings import get_embedding_function
from langchain_chroma import Chroma
from langchain.schema.document import Document


CHROMA_PATH = "chroma"


# Storing information using embeddings "vector" as a key
def add_to_chroma(chunks: list[Document], EmbModel):
    # Load the existing DB
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function(EmbModel),
    )
    db._client._system.start()
    try:
        # Calculate Page IDs
        chunks_with_ids = calculate_chunk_ids(chunks)

        # ADD or Update the documents
        existing_items = db.get(include=[])  # IDs are always included by default
        existing_ids = set(existing_items["ids"])
        print(f"Number of existing documents in DB: {len(existing_ids)}")
        # Only add documents that don't exist
        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        if len(new_chunks):
            print(f"👉 Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
            print(f"✅ Documents Added To Chroma")
        else:
            print("✅ No new documents to add")
    finally:
        db._client._system.stop()


def calculate_chunk_ids(chunks):
    # This will create IDs like "data\\books\\file.pdf:0:0"
    # Page Source : Page Number : Chunk Index
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page") or chunk.metadata.get("page_number")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data
        chunk.metadata["id"] = chunk_id
    return chunks


def clear_database(selected_files):
    try:
        # Initialize Chroma instance
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=None)
        db._client._system.start()

        # Construct the list of source paths to match
        files_paths = [f"Data\\Documents\\{name}" for name in selected_files]

        # Delete documents where the "source" metadata matches any of the selected files
        db.delete(
            where={  # Filter by metadata
                "source": {  # Replace "source" with the actual metadata key if different
                    "$in": files_paths
                }
            }
        )

        print(
            f"✅ Documents from selected files deleted successfully: \n{"\n".join(selected_files)}"
        )
    except Exception as e:
        print(f"❌ Error clearing database: {e}")
    finally:
        db._client._system.stop()
