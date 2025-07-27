"""
This module defines the VectorDB class, which manages the creation, loading,
and querying of vector stores using ChromaDB.
"""

import json
import os
from typing import Any, Dict, List, Optional

from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from source.config import (
    NUMBER_OF_RETRIEVED_CHUNKS_LIMIT,
    VECTOR_DATA_PATH,
)
from source.core.embeddings import Embeddings


class VectorDB:
    """
    Manages interactions with ChromaDB vector stores.

    This class handles the entire lifecycle of vector data:
    1.  Creating document chunks and summaries into LangChain `Document` objects.
    2.  Creating and persisting ChromaDB vector stores for each source file.
    3.  Loading existing vector stores from disk on initialization.
    4.  Providing a retriever interface to query the vector stores.

    Attributes:
        chunks (Optional[Dict[str, Any]]): A dictionary containing the chunked
            data to be added to the vector store.
        vector_data_path (str): The root directory for persisting vector stores.
        num_of_retrieved_chunks (int): The default number of chunks to retrieve.
        id_to_file_dict (Dict[str, str]): Mapping from unique ID to filename.
        file_to_id_dict (Dict[str, str]): Mapping from filename to unique ID.
        embedding_model (Embeddings): The model used to generate embeddings.
        vectorstores (Dict[str, Chroma]): A dictionary holding loaded vector
            store instances.
    """

    def __init__(
        self,
        chunks: Optional[Dict[str, Any]] = None,
        num_of_retrieved_chunks: int = NUMBER_OF_RETRIEVED_CHUNKS_LIMIT,
        vector_data_path: str = VECTOR_DATA_PATH,
    ) -> None:
        """
        Initializes the VectorDB class.

        Args:
            chunks (Optional[Dict[str, Any]]): Data to be vectorized.
            num_of_retrieved_chunks (int): Default 'k' value for retrieval.
            vector_data_path (str): Path to the persistent vector data directory.
        """
        self.chunks: Optional[Dict[str, Any]] = chunks
        self.vector_data_path: str = vector_data_path
        self.num_of_retrieved_chunks: int = num_of_retrieved_chunks
        
        try:
            with open("./id_to_file_map.json", "r") as f:
                id_to_file_map: Dict[str, str] = json.load(f)
            self.id_to_file_dict: Dict[str, str] = id_to_file_map
            self.file_to_id_dict: Dict[str, str] = {v: k for k, v in id_to_file_map.items()}
        except FileNotFoundError:
            print("Error: 'id_to_file_map.json' not found. Please ensure it exists.")
            self.id_to_file_dict = {}
            self.file_to_id_dict = {}

        self.embedding_model: Embeddings = Embeddings()
        self.vectorstores: Dict[str, Chroma] = {}
        self._load_chroma_vectorstores()

    def create_vector_stores(self) -> None:
        """
        Orchestrates the creation of vector stores for all provided chunks.
        """
        if not self.chunks:
            print("No chunks provided to create vector stores.")
            return
            
        for filename, chunks_dict in self.chunks.items():
            documents = self._create_docs_from_chunks(filename, chunks_dict["chunks"])
            self._create_vector_store(filename, documents)
        self._create_document_summary_vector_store()

    def get_retriever(
        self, file_name: str, k: Optional[int] = None
    ) -> VectorStoreRetriever:
        """
        Gets a retriever for a specific vector store collection.

        Args:
            file_name (str): The name of the file whose collection is needed.
            k (Optional[int]): The number of documents to retrieve. Defaults to
                               `self.num_of_retrieved_chunks`.

        Returns:
            VectorStoreRetriever: A LangChain retriever instance.
        """
        collection_name = (
            f"Collection_{self.file_to_id_dict[file_name]}"
            if file_name in self.file_to_id_dict
            else file_name
        )
        vectorstore = self.vectorstores[collection_name]
        k_final = k if k is not None else self.num_of_retrieved_chunks
        
        return vectorstore.as_retriever(
            search_type="mmr", search_kwargs={"k": k_final}
        )

    def _create_document_summary_vector_store(self) -> None:
        """Creates a single vector store for the summaries of all documents."""
        summary_dict = {
            filename: chunks_dict["doc_summary"]
            for filename, chunks_dict in self.chunks.items()
        }
        documents = self._create_docs_from_summary(summary_dict)
        self._create_vector_store("all_docs_summary", documents)

    def _create_vector_store(self, file_name: str, documents: List[Document]) -> None:
        """Creates or updates a Chroma vector store for a given file."""
        collection_name = (
            f"Collection_{self.file_to_id_dict[file_name]}"
            if file_name in self.file_to_id_dict
            else file_name
        )
        persist_dir = os.path.join(self.vector_data_path, collection_name)

        if os.path.exists(persist_dir):
            # Load existing collection and add new documents
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=self.embedding_model,
                persist_directory=persist_dir,
            )
            vectorstore.add_documents(documents)
        else:
            # Create a new collection from documents
            vectorstore = Chroma.from_documents(
                documents=documents,
                collection_name=collection_name,
                embedding=self.embedding_model,
                persist_directory=persist_dir,
            )
        vectorstore.persist()
        self.vectorstores[collection_name] = vectorstore

    def _load_chroma_vectorstores(self) -> None:
        """Loads all existing vector stores from the persist directory."""
        if not os.path.isdir(self.vector_data_path):
            return

        for folder_name in os.listdir(self.vector_data_path):
            folder_path = os.path.join(self.vector_data_path, folder_name)
            if not os.path.isdir(folder_path):
                continue
            try:
                vectorstore = Chroma(
                    persist_directory=folder_path,
                    collection_name=folder_name,
                    embedding_function=self.embedding_model,
                )
                self.vectorstores[folder_name] = vectorstore
            except Exception as e:
                print(f"Error loading vectorstore from {folder_name}: {e}")

    @staticmethod
    def _create_docs_from_summary(summary_dict: Dict[str, str]) -> List[Document]:
        """Creates a list of Document objects from a dictionary of summaries."""
        return [
            Document(page_content=summary, metadata={"filename": filename})
            for filename, summary in summary_dict.items()
        ]

    @staticmethod
    def _create_docs_from_chunks(
        filename: str, chunks_dict: Dict[str, Any]
    ) -> List[Document]:
        """Creates a list of Document objects from a dictionary of chunks."""
        return [
            Document(
                page_content=" ".join(chunk["propositions"]),
                metadata={
                    "chunk_id": chunk_id,
                    "source": filename,
                    "summary": chunk["summary"],
                    "title": chunk["title"],
                },
            )
            for chunk_id, chunk in chunks_dict.items()
        ]
