"""
This module defines the Backend class, which serves as the main orchestrator
for the EcoAI RAG (Retrieval-Augmented Generation) pipeline.
"""

from collections import defaultdict
from typing import Any, Dict, List, Tuple

from source.core.generator import Generator
from source.core.retriever import Retriever


class Backend:
    """
    The Backend class orchestrates the entire question-answering process.

    It initializes the `Retriever` and `Generator` components and manages the
    flow of data between them. It maintains a conversation history and uses it,
    along with retrieved documents, to generate context-aware answers.

    Attributes:
        retriever (Retriever): The component responsible for fetching relevant
                               documents from the vector store.
        generator (Generator): The component responsible for generating answers
                               based on the context.
        messages (defaultdict[str, List[Any]]): A dictionary holding the
                                                conversation history, separated
                                                by sender ('User' or 'AI').
    """

    def __init__(self, load_docs: bool = False) -> None:
        """
        Initializes the Backend class.

        Args:
            load_docs (bool): A flag passed to the Retriever to determine
                              whether to load documents into memory on
                              initialization. Defaults to False.
        """
        self.retriever: Retriever = Retriever(load_docs)
        self.generator: Generator = Generator()
        self.messages: Dict[str, List[Any]] = defaultdict(list)

    def _append_message(self, message: Any, sender: str) -> None:
        """
        Appends a message to the conversation history.

        Args:
            message (Any): The content of the message.
            sender (str): The sender of the message, typically 'User' or 'AI'.
        """
        self.messages[sender].append(message)

    def run(self, query: str) -> Tuple[str, float, List[str]]:
        """
        Executes the main RAG pipeline for a given user query.

        This method performs the following steps:
        1. Appends the user's query to the conversation history.
        2. Uses the retriever to get the most relevant documents, along with
           a confidence score and source information.
        3. Passes the conversation history and retrieved documents to the
           generator to produce an answer.
        4. Appends the AI's response to the conversation history.
        5. Returns the final response, confidence score, and sources.

        Args:
            query (str): The user's question.

        Returns:
            Tuple[str, float, List[str]]: A tuple containing the generated
                                          answer, the retrieval confidence
                                          score, and a list of source documents.
        """
        self._append_message(query, "User")
        reranked_docs, confidence, sources = self.retriever.get_docs(query)
        response = self.generator.generate(self.messages, reranked_docs)
        self._append_message(response, "AI")
        return response, confidence, sources