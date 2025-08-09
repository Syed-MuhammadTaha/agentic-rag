import requests
from config import Config
from typing import List
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel

class Embedder(BaseModel, Embeddings):
    """
    A class to generate text embeddings using TorchServe.
    """
    @classmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Generates an embedding for a single query text using TorchServe.
        """
        headers = {"Content-Type": "application/json"}
        data = {"input": [text]}

        try:
            response = requests.post(Config.TORCHSERVE_URL, json=data, headers=headers)
            response.raise_for_status()
            return response.json()[0]
        except Exception as e:
            print(f"Error generating embedding for query: {e}")
            return None

    @classmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for multiple documents using TorchServe.
        """
        headers = {"Content-Type": "application/json"}
        data = {"input": texts}

        try:
            response = requests.post(Config.TORCHSERVE_URL, json=data, headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error generating embeddings for documents: {e}")
            return None


if __name__ == "__main__":
    print(len(Embedder.embed_query("Hello, world!")))
    print(len(Embedder.embed_documents(["Hello, world!", "Hello, world!"])))