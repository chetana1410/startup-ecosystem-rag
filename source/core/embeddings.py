"""
This module defines the Embeddings class, which is a wrapper around a
SentenceTransformer model to generate vector embeddings for text.
"""

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from source.config import EMBEDDING_MODEL_NAME


class Embeddings:
    """
    Handles the generation of text embeddings using a SentenceTransformer model.

    This class abstracts the complexity of encoding texts into dense vectors.
    It provides simple methods to embed single queries or batches of documents,
    ensuring that the embeddings are normalized and returned in a consistent format.

    Attributes:
        embedding_model (SentenceTransformer): The loaded SentenceTransformer model instance.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        """
        Initializes the Embeddings class and loads the specified model.

        Args:
            model_name (str): The name of the SentenceTransformer model to load.
                              Defaults to the model specified in the config.
        """
        try:
            self.embedding_model: SentenceTransformer = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Error loading SentenceTransformer model '{model_name}': {e}")
            # Depending on the application's needs, you might want to exit
            # or handle this more gracefully.
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Encodes a list of document texts into a list of vector embeddings.

        Args:
            texts (List[str]): A list of strings, where each string is a
                               document or text chunk to be embedded.

        Returns:
            List[List[float]]: A list of embeddings, where each embedding is a
                               list of floats.
        """
        embeddings: np.ndarray = self.embedding_model.encode(
            texts, normalize_embeddings=True
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Encodes a single query text into a vector embedding.

        Args:
            text (str): The query string to embed.

        Returns:
            List[float]: The vector embedding for the query.
        """
        embedding: np.ndarray = self.embedding_model.encode(
            [text], normalize_embeddings=True
        )
        return embedding.tolist()[0]
