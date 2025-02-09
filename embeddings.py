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