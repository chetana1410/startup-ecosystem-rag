"""
This module defines the Retriever class, which is responsible for fetching
and re-ranking relevant document chunks from the vector store.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import prod
from typing import Any, List, Set, Tuple

import torch
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from sentence_transformers import CrossEncoder

from source.config import (
    DATA_FOLDER_PATH,
    NUMBER_OF_TOP_RERANK_CHUNKS,
    RERANKER_MODEL_NAME,
    RERANK_SCORE_DIFFERECE_THRESHOLD,
)
from source.core.doc_loader import DocLoader
from source.core.propositions import ExtractPropositions
from source.core.vectordb import VectorDB


class Retriever:
    """
    Handles the retrieval of relevant documents from multiple vector stores
    and re-ranks them to find the best context for a given query.

    This class orchestrates the following steps:
    1. Optionally loads and processes source documents if they are not already
       vectorized.
    2. Concurrently queries the vector store for each document to get
       initial candidate chunks.
    3. Uses a CrossEncoder model to re-rank the retrieved chunks based on
       their semantic similarity to the query.
    4. Calculates a confidence score based on the re-ranked results.
    5. Returns the top-ranked documents, confidence score, and source filenames.

    Attributes:
        load_docs (bool): Flag to indicate whether to run the data loading pipeline.
        reranker (CrossEncoder): The CrossEncoder model for re-ranking documents.
        vector_db (VectorDB): The interface to the ChromaDB vector stores.
        folder_path (str): The path to the folder containing source documents.
    """

    def __init__(self, load_docs: bool = False) -> None:
        """
        Initializes the Retriever class.

        Args:
            load_docs (bool): If True, triggers the document loading and
                              processing pipeline. Defaults to False.
        """
        self.load_docs: bool = load_docs
        self.reranker: CrossEncoder = CrossEncoder(RERANKER_MODEL_NAME, device="mps")
        self.vector_db: VectorDB = VectorDB()
        self.folder_path: str = DATA_FOLDER_PATH
        self._load_data()

    def get_docs(self, query: str) -> Tuple[List[Tuple[float, Document]], float, str]:
        """
        Retrieves, re-ranks, and returns the most relevant documents for a query.

        Args:
            query (str): The user's question.

        Returns:
            Tuple[List[Tuple[float, Document]], float, str]: A tuple containing:
                - A list of (score, Document) tuples for the top-ranked documents.
                - A confidence score for the retrieval.
                - A comma-separated string of source filenames.
        """
        # Get a list of all PDF files to query against
        try:
            files = [f for f in os.listdir(self.folder_path) if f.endswith(".pdf")]
        except FileNotFoundError:
            print(f"Error: The data directory '{self.folder_path}' was not found.")
            return [], 0.0, "Error: Data directory not found."

        all_docs: List[Document] = []
        # Concurrently retrieve chunks from each file's vector store
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_file = {
                executor.submit(self._get_chunks_from_doc, file, query): file
                for file in files
            }
            for future in as_completed(future_to_file):
                try:
                    retrieved_docs = future.result()
                    all_docs.extend(retrieved_docs)
                except Exception as e:
                    filename = future_to_file[future]
                    print(f"Error retrieving chunks from {filename}: {e}")

        if not all_docs:
            return [], 0.0, "No relevant documents found."

        # Re-rank all retrieved documents to find the best ones
        reranked_docs = self._rerank_documents(
            query, all_docs, top_k=NUMBER_OF_TOP_RERANK_CHUNKS
        )
        
        # Calculate a confidence score as the product of the top scores
        confidence = prod([score for score, _ in reranked_docs])

        # Collect unique source filenames from the top documents
        sources: Set[str] = {doc.metadata["source"] for _, doc in reranked_docs}

        return reranked_docs, confidence, ", ".join(sorted(list(sources)))

    def _load_data(self) -> None:
        """
        Initiates the data loading and processing pipeline if `load_docs` is True.
        """
        if self.load_docs:
            print("Starting document loading and processing pipeline...")
            doc_loader = DocLoader(DATA_FOLDER_PATH)
            all_docs_text = doc_loader.load_docs()

            if all_docs_text:
                extract_propositions = ExtractPropositions(all_docs_text)
                extract_propositions.extract()
            else:
                print("No new documents to load.")

    def _get_chunks_from_doc(self, file_name: str, query: str) -> List[Document]:
        """
        Retrieves document chunks for a single file from its vector store.

        Args:
            file_name (str): The name of the file to retrieve chunks from.
            query (str): The user query.

        Returns:
            List[Document]: A list of retrieved Document objects.
        """
        retriever: VectorStoreRetriever = self.vector_db.get_retriever(file_name)
        return retriever.invoke(query)

    def _rerank_documents(
        self, query: str, documents: List[Document], top_k: int = 5
    ) -> List[Tuple[float, Document]]:
        """
        Re-ranks a list of documents based on their relevance to the query.

        Args:
            query (str): The user query.
            documents (List[Document]): The list of documents to re-rank.
            top_k (int): The maximum number of documents to return.

        Returns:
            List[Tuple[float, Document]]: A sorted list of (score, Document) tuples.
        """
        if not documents:
            return []
            
        # Prepare pairs of [query, document_content] for the CrossEncoder
        pairs: List[List[str]] = [[query, doc.page_content] for doc in documents]

        # Predict relevance scores
        scores: List[float] = self.reranker.predict(pairs)

        # Combine scores with documents and sort
        sorted_indices = torch.argsort(torch.tensor(scores), descending=True)
        sorted_docs_with_scores = [
            (scores[i], documents[i]) for i in sorted_indices
        ]

        # Select top documents, applying a score difference threshold
        selected_docs: List[Tuple[float, Document]] = []
        prev_score: float = None
        for score, doc in sorted_docs_with_scores:
            if prev_score is not None and (prev_score - score) >= RERANK_SCORE_DIFFERECE_THRESHOLD:
                break  # Stop if the score drops significantly
            selected_docs.append((score, doc))
            prev_score = score
            if len(selected_docs) >= top_k:
                break  # Stop once top_k is reached

        return selected_docs
