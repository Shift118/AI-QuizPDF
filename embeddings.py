from langchain_ollama import OllamaEmbeddings
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def get_embedding_function(EmbModel: str):
    if EmbModel == "Gemini API":

        class GeminiEmbeddings:
            def __init__(self, api_key: str, model: str = "models/text-embedding-004"):
                genai.configure(api_key=api_key)
                self.model = model

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                embeddings = []
                for text in texts:
                    result = genai.embed_content(
                        model=self.model, content=text, task_type="RETRIEVAL_DOCUMENT"
                    )
                    # print("Document embedding result:", result)
                    # print("Document embedding type:", type(result))
                    # Extract the embedding from the dictionary
                    embeddings.append(result["embedding"])  # Access the 'embedding' key
                return embeddings

            def embed_query(self, text: str) -> list[float]:
                result = genai.embed_content(
                    model=self.model, content=text, task_type="RETRIEVAL_QUERY"
                )
                # print("Query embedding result:", result)
                # print("Query embedding type:", type(result))
                # Extract the embedding from the dictionary
                return result["embedding"]  # Access the 'embedding' key

        return GeminiEmbeddings(api_key=GEMINI_API_KEY)

    if EmbModel == "Nomic-Embed-Text":
        return OllamaEmbeddings(model="nomic-embed-text", num_thread=3)


'''
from langchain_ollama import OllamaEmbeddings
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def get_embedding_function(EmbModel: str):
    if EmbModel == "Gemini API":
        class GeminiEmbeddings:
            def __init__(self, api_key: str, model: str = "models/text-embedding-004"):
                genai.configure(api_key=api_key)
                self.model = model

            def _get_gemini_embedding(self, text: str, task_type: str) -> list[float] | None:  # Helper function
                """Gets embedding from Gemini and handles potential errors."""
                try:
                    result = genai.embed_content(
                        model=self.model,
                        content=text,
                        task_type=task_type
                    )
                    print(f"{task_type.split('_')[1].lower()} embedding result: {result}") # Nicer print output
                    print(f"{task_type.split('_')[1].lower()} embedding type: {type(result)}")
                    return result.get('embedding')  # Safer access; returns None if 'embedding' is missing
                except Exception as e: # Catch potential errors and return None
                    print(f"Error getting embedding: {e}")
                    return None

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                embeddings = []
                for text in texts:
                    embedding = self._get_gemini_embedding(text, "RETRIEVAL_DOCUMENT")
                    if embedding:  # Check if embedding was successfully retrieved
                        embeddings.append(embedding)
                    else:
                        print(f"Warning: Could not get embedding for document: '{text[:50]}...'") # Print a warning for missing embeddings
                return embeddings

            def embed_query(self, text: str) -> list[float]:
                embedding = self._get_gemini_embedding(text, "RETRIEVAL_QUERY")
                if embedding:
                    return embedding
                else:
                    print(f"Error: Could not get embedding for query: '{text[:50]}...'")
                    return [] # Return empty list in case of error, to prevent further errors.

        return GeminiEmbeddings(api_key=GEMINI_API_KEY)

    if EmbModel == "Nomic-Embed-Text":
        return OllamaEmbeddings(
            model="nomic-embed-text",
            num_thread=3
        )
'''
'''import chromadb
import google.generativeai as genai
import os

# --- Configuration ---
CHROMA_PATH = "./chroma_db"  # Path to store Chroma database
EmbModel = "Gemini-API"  # Choose embedding model: "Gemini-API" or "Nomic-Embed-Text" (placeholder)

# --- Gemini API Embedding Function ---
def generate_gemini_embedding(text):
    """Generates a single embedding for the given text using the Gemini API."""
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise EnvironmentError(
            "GOOGLE_API_KEY environment variable not set. "
            "Please set your Gemini API key as an environment variable."
        )
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    response = model.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_query",  # Or adjust task_type as needed: "retrieval_document", "semantic_similarity"
        title="Embedding for text: " + text
    )
    return response.embedding.values

def GeminiAPIEmbeddings():
    """
    Returns an embedding function that uses the Gemini API, compatible with Chroma.
    """
    def embed_documents(texts):
        """Embeds a list of documents using the Gemini API."""
        return [generate_gemini_embedding(text) for text in texts]
    return embed_documents

# --- Placeholder for Ollama Embeddings (Replace with your actual Ollama setup if needed) ---
def OllamaEmbeddingsPlaceholder():
    """
    Placeholder for Ollama embeddings.
    Replace this with your actual OllamaEmbeddings implementation if you want to use it.
    """
    def embed_documents(texts):
        """Placeholder embedding function - replace with actual Ollama logic."""
        print("Using placeholder Ollama embedding function (replace with your Ollama setup!)")
        return [[0.0] * 10 for _ in texts]  # Dummy embeddings for demonstration
    return embed_documents


# --- Function to dynamically get embedding function based on EmbModel ---
def get_embedding_function(emb_model):
    if emb_model == "Nomic-Embed-Text":
        # Replace OllamaEmbeddingsPlaceholder() with your actual OllamaEmbeddings setup here
        return OllamaEmbeddingsPlaceholder() # <--- Replace this with your actual OllamaEmbeddings() function if using Ollama
    elif emb_model == "Gemini-API":
        return GeminiAPIEmbeddings()
    else:
        raise ValueError(f"Unsupported embedding model: {emb_model}")

# --- Initialize ChromaDB ---
db = chromadb.PersistentClient(path=CHROMA_PATH) # Use PersistentClient for persistence
embedding_function = get_embedding_function(EmbModel)

collection = db.get_or_create_collection(
    "my_gemini_collection", # Choose a descriptive collection name
    embedding_function=embedding_function
)

print(f"Chroma database initialized with {EmbModel} embeddings.")

# --- Example Usage: Adding Documents ---
documents = [
    "The quick brown rabbit jumps over the lazy frogs.",
    "Artificial intelligence is rapidly transforming many industries.",
    "My favorite color is blue because it reminds me of the ocean.",
    "Summer is the best season because of the warm weather and long days."
]
ids = ["doc1", "doc2", "doc3", "doc4"] # Assign unique IDs

collection.add(
    documents=documents,
    ids=ids
)
print("Documents added to collection.")

# --- Example Usage: Querying ---
query_text = "What are the benefits of artificial intelligence?"
query_results = collection.query(
    query_texts=[query_text],
    n_results=2 # Get top 2 most relevant results
)

print(f"\nQuery: '{query_text}'")
print("Query Results:")
print(query_results)

print("\nChroma database operations complete.")

# --- Important Reminders ---
print("\n--- Important Reminders ---")
print("1. Set GOOGLE_API_KEY environment variable: "
      "Before running, ensure you have set your GOOGLE_API_KEY environment variable "
      "with your actual Gemini API key from Google Cloud Console.")
print("2. Install required libraries: "
      "Make sure you have installed chromadb and google-generativeai libraries. "
      "Run: `pip install chromadb google-generativeai`")
print("3. Ollama Setup (if using Nomic-Embed-Text): "
      "If you intend to use 'Nomic-Embed-Text', you need to replace the OllamaEmbeddingsPlaceholder() "
      "with your actual Ollama setup and ensure Ollama is running with the 'nomic-embed-text' model available.")'''
