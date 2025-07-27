"""
This module defines the AgenticChunker class, a core component of the EcoAI
RAG pipeline responsible for intelligently grouping propositions into coherent
chunks.
"""

from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from ollama import chat
from pydantic import BaseModel  # <-- ADDED THIS IMPORT
from tqdm import tqdm

from source.config import (
    CHUNK_ID_TRUNCATE_LIMIT,
    CHUNK_SIZE_LIMIT,
    LLM_MODEL_NAME,
)
from source.core.vectordb import VectorDB
from source.utils import (
    DOC_SUMMARY_SYSTEM_PROMPT,
    FIND_RELEVANT_CHUNK_SYSTEM_PROMPT,
    NEW_CHUNK_SUMMARY_SYSTEM_PROMPT,
    NEW_CHUNK_TITLE_SYSTEM_PROMPT,
    UPDATE_CHUNK_SUMMARY_SYSTEM_PROMPT,
    UPDATE_CHUNK_TITLE_SYSTEM_PROMPT,
    ChunkID,
    Summary,
    Title,
)

# A type alias for a dictionary representing a single chunk.
Chunk = Dict[str, Union[str, List[str], int]]


class AgenticChunker:
    """
    An intelligent chunker that uses an LLM to group propositions into
    semantically coherent chunks.

    This class takes a dictionary of propositions, where keys are filenames
    and values are lists of propositions (strings). It then iteratively
    processes each proposition, deciding whether to add it to an existing
    chunk or create a new one. This decision is made by an LLM, which also
    helps in generating and updating titles and summaries for each chunk.

    Attributes:
        propositions_dict (Dict[str, List[str]]): The input propositions.
        model_name (str): The name of the Ollama model to use.
        chunk_size_limit (int): The maximum character length for a chunk's propositions.
        id_truncate_limit (int): The character limit for generated chunk IDs.
        print_logging (bool): If True, prints detailed logs during processing.
        generate_new_metadata_ind (bool): If True, updates chunk titles and
                                          summaries as new propositions are added.
        all_chunks (Dict[str, Any]): A dictionary to store the final chunked
                                     output for all processed files.
    """

    def __init__(
        self,
        propositions_dict: Dict[str, List[str]],
        model_name: str = LLM_MODEL_NAME,
        chunk_size_limit: int = CHUNK_SIZE_LIMIT,
        id_truncate_limit: int = CHUNK_ID_TRUNCATE_LIMIT,
        print_logging: bool = False,
        generate_new_metadata_ind: bool = True,
    ) -> None:
        """
        Initializes the AgenticChunker.

        Args:
            propositions_dict (Dict[str, List[str]]): A dictionary mapping filenames
                to lists of propositions.
            model_name (str): The Ollama model name.
            chunk_size_limit (int): The maximum character size for a chunk.
            id_truncate_limit (int): The length of the UUID for chunk IDs.
            print_logging (bool): Flag to enable/disable detailed logging.
            generate_new_metadata_ind (bool): Flag to enable/disable metadata
                updates on-the-fly.
        """
        self.propositions_dict: Dict[str, List[str]] = propositions_dict
        self.id_truncate_limit: int = id_truncate_limit
        self.chunk_size_limit: int = chunk_size_limit
        self.generate_new_metadata_ind: bool = generate_new_metadata_ind
        self.print_logging: bool = print_logging
        self.model: str = model_name

        # System prompts for various LLM tasks
        self.update_chunk_summary_system_prompt: str = (
            UPDATE_CHUNK_SUMMARY_SYSTEM_PROMPT
        )
        self.update_chunk_title_system_prompt: str = UPDATE_CHUNK_TITLE_SYSTEM_PROMPT
        self.new_chunk_summary_system_prompt: str = NEW_CHUNK_SUMMARY_SYSTEM_PROMPT
        self.new_chunk_title_system_prompt: str = NEW_CHUNK_TITLE_SYSTEM_PROMPT
        self.find_relevant_chunk_system_prompt: str = (
            FIND_RELEVANT_CHUNK_SYSTEM_PROMPT
        )
        self.doc_summary_system_prompt: str = DOC_SUMMARY_SYSTEM_PROMPT

        self.all_chunks: Dict[str, Dict[str, Any]] = {}
        self.chunks: Dict[str, Chunk] = {}

    def chunk_docs(self) -> None:
        """
        The main method to process all documents in the propositions_dict.

        It iterates through each file's propositions, chunks them, generates a
        document summary, and creates a vector store for the processed file.
        """
        for filename, propositions in self.propositions_dict.items():
            print(f"Chunking file: {filename}")
            self.chunks = {}
            self._add_propositions(propositions)
            summary = self._get_doc_summary()

            processed_file_data = {"chunks": self.chunks, "doc_summary": summary}
            self.all_chunks[filename] = processed_file_data

            print(processed_file_data)
            vector_db = VectorDB({filename: processed_file_data})
            vector_db.create_vector_stores()

    def _add_propositions(self, propositions: List[str]) -> None:
        """
        Iterates through a list of propositions and adds each one to a chunk.

        Args:
            propositions (List[str]): The list of propositions for a single document.
        """
        for proposition in tqdm(
            propositions, total=len(propositions), desc="Adding propositions"
        ):
            self._add_proposition(proposition)

    def _add_proposition(self, proposition: str) -> None:
        """
        Processes a single proposition to find a relevant chunk or create a new one.

        Args:
            proposition (str): The proposition to be added.
        """
        if self.print_logging:
            print(f"Adding: '{proposition}'")

        if not self.chunks:
            if self.print_logging:
                print("No chunks exist, creating a new one.")
            self._create_new_chunk(proposition)
            return

        # Retry logic to ensure a valid chunk_id is processed
        while True:
            chunk_id = self._find_relevant_chunk(proposition)
            if chunk_id and chunk_id not in self.chunks:
                continue  # The LLM might hallucinate a non-existent ID, so we retry.
            break

        if chunk_id:
            # Check if the chunk has space
            current_chunk_text = "".join(self.chunks[chunk_id]["propositions"])
            if len(current_chunk_text) < self.chunk_size_limit:
                if self.print_logging:
                    print(
                        f"Chunk Found ({chunk_id}), adding to: {self.chunks[chunk_id]['title']}"
                    )
                self.add_proposition_to_chunk(chunk_id, proposition)
            else:
                if self.print_logging:
                    print(
                        f"Chunk Found ({chunk_id}) but max size reached, creating a new one."
                    )
                self._create_new_chunk(proposition)
        else:
            if self.print_logging:
                print("No relevant chunk found, creating a new one.")
            self._create_new_chunk(proposition)

    def add_proposition_to_chunk(self, chunk_id: str, proposition: str) -> None:
        """
        Adds a proposition to a specified chunk and updates its metadata.

        Args:
            chunk_id (str): The ID of the chunk to add the proposition to.
            proposition (str): The proposition to add.
        """
        self.chunks[chunk_id]["propositions"].append(proposition)

        if self.generate_new_metadata_ind:
            self.chunks[chunk_id]["summary"] = self._update_chunk_summary(
                self.chunks[chunk_id]
            )
            self.chunks[chunk_id]["title"] = self._update_chunk_title(
                self.chunks[chunk_id]
            )

    def _llm_call(
        self, system_prompt: str, user_content: str, response_model: BaseModel
    ) -> Any:
        """A helper function to make calls to the Ollama chat API."""
        response = chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            model=self.model,
            format=response_model.model_json_schema(),
        )
        return response_model.model_validate_json(response.message.content)

    def _update_chunk_summary(self, chunk: Chunk) -> str:
        """Updates a chunk's summary based on its current propositions."""
        propositions = "\n".join(chunk["propositions"])
        user_content = f"Chunk's propositions:\n{propositions}\n\nCurrent chunk summary:\n{chunk['summary']}"
        response: Summary = self._llm_call(
            self.update_chunk_summary_system_prompt, user_content, Summary
        )
        return response.summary

    def _update_chunk_title(self, chunk: Chunk) -> str:
        """Updates a chunk's title based on its current propositions and summary."""
        propositions = "\n".join(chunk["propositions"])
        user_content = f"Chunk's propositions:\n{propositions}\n\nChunk summary:\n{chunk['summary']}\n\nCurrent chunk title:\n{chunk['title']}"
        response: Title = self._llm_call(
            self.update_chunk_title_system_prompt, user_content, Title
        )
        return response.title

    def _get_new_chunk_summary(self, proposition: str) -> str:
        """Generates a summary for a new chunk based on its first proposition."""
        user_content = f"Determine the summary of the new chunk that this proposition will go into:\n{proposition}"
        response: Summary = self._llm_call(
            self.new_chunk_summary_system_prompt, user_content, Summary
        )
        return response.summary

    def _get_new_chunk_title(self, summary: str) -> str:
        """Generates a title for a new chunk based on its summary."""
        user_content = (
            f"Determine the title of the chunk that this summary belongs to:\n{summary}"
        )
        response: Title = self._llm_call(
            self.new_chunk_title_system_prompt, user_content, Title
        )
        return response.title

    def _create_new_chunk(self, proposition: str) -> None:
        """Creates a new chunk with a unique ID, summary, and title."""
        new_chunk_id = str(uuid4())[: self.id_truncate_limit]
        new_chunk_summary = self._get_new_chunk_summary(proposition)
        new_chunk_title = self._get_new_chunk_title(new_chunk_summary)

        self.chunks[new_chunk_id] = {
            "chunk_id": new_chunk_id,
            "propositions": [proposition],
            "title": new_chunk_title,
            "summary": new_chunk_summary,
            "chunk_index": len(self.chunks),
        }
        if self.print_logging:
            print(f"Created new chunk ({new_chunk_id}): {new_chunk_title}")

    def get_chunk_outline(self) -> str:
        """Generates a formatted string outline of all current chunks."""
        return "\n\n".join(
            [
                f"Chunk ({chunk['chunk_id']}): {chunk['title']}\nSummary: {chunk['summary']}"
                for chunk in self.chunks.values()
            ]
        )

    def _find_relevant_chunk(self, proposition: str) -> Optional[str]:
        """Asks the LLM to find the most relevant chunk for a given proposition."""
        current_chunk_outline = self.get_chunk_outline()
        user_content = f"""
        Current Chunks:
        --Start of current chunks--
        {current_chunk_outline}
        --End of current chunks--

        Determine if the following statement should belong to one of the chunks outlined:
        {proposition}
        """
        response: ChunkID = self._llm_call(
            self.find_relevant_chunk_system_prompt, user_content, ChunkID
        )

        if not response.chunk_id or len(response.chunk_id) != self.id_truncate_limit:
            return None
        return response.chunk_id

    def _get_doc_summary(self) -> str:
        """Generates a summary for the entire document from its chunk summaries."""
        combined_summaries = "\n".join(
            [chunk["summary"] for chunk in self.chunks.values()]
        )

        if self.print_logging:
            print("Combined Chunk Summary:")
            print(combined_summaries)
            print(f"Length of combined chunk summary: {len(combined_summaries)}")

        user_content = f"Summary:\n{combined_summaries}"
        response: Summary = self._llm_call(
            self.doc_summary_system_prompt, user_content, Summary
        )
        return response.summary

    def get_chunks(self, get_type: str = "dict") -> Union[Dict[str, Chunk], List[str]]:
        """
        Returns the processed chunks in a specified format.

        Args:
            get_type (str): The desired format ('dict' or 'list_of_strings').

        Returns:
            Union[Dict[str, Chunk], List[str]]: The chunks in the requested format.
        """
        if get_type == "list_of_strings":
            return [" ".join(chunk["propositions"]) for chunk in self.chunks.values()]
        return self.chunks

    def pretty_print_chunks(self) -> None:
        """Prints a detailed, human-readable view of all chunks."""
        print(f"\nYou have {len(self.chunks)} chunks\n")
        for chunk_id, chunk in self.chunks.items():
            print(f"Chunk #{chunk['chunk_index']}")
            print(f"Chunk ID: {chunk_id}")
            print(f"Summary: {chunk['summary']}")
            print("Propositions:")
            for prop in chunk["propositions"]:
                print(f"    -{prop}")
            print("\n\n")

    def pretty_print_chunk_outline(self) -> None:
        """Prints the formatted chunk outline to the console."""
        print("Chunk Outline\n")
        print(self.get_chunk_outline())
