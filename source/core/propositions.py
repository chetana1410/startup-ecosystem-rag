"""
This module defines the ExtractPropositions class, which is responsible for
decomposing raw text from documents into a list of clear, atomic propositions
using a Language Model.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from ollama import chat
from tqdm import tqdm

from source.config import LLM_MODEL_NAME
from source.core.agentic_chunker import AgenticChunker
from source.utils import PROPOSITIONS_SYSTEM_PROMPT, Sentences


class ExtractPropositions:
    """
    Extracts atomic propositions from text files using an LLM.

    This class orchestrates the process of breaking down large text documents
    into smaller, more manageable paragraphs, and then using a multi-threaded
    approach to call an LLM to decompose each paragraph into a list of simple,
    context-independent propositions.

    Attributes:
        files (Dict[str, str]): A dictionary mapping filenames to their raw
                                text content.
        model_name (str): The name of the Ollama model to use for proposition
                          extraction.
        system_prompt (str): The system prompt guiding the LLM's decomposition task.
        propositions (Dict[str, List[str]]): A dictionary to store the extracted
                                             propositions for each file.
    """

    def __init__(self, files: Dict[str, str], model_name: str = LLM_MODEL_NAME) -> None:
        """
        Initializes the ExtractPropositions class.

        Args:
            files (Dict[str, str]): A dictionary where keys are filenames and
                                   values are the full text content of the files.
            model_name (str): The name of the Ollama model to use.
        """
        self.files: Dict[str, str] = files
        self.model_name: str = model_name
        self.system_prompt: str = PROPOSITIONS_SYSTEM_PROMPT
        self.propositions: Dict[str, List[str]] = {}

    def _get_propositions_from_paragraph(self, paragraph: str) -> List[str]:
        """
        Sends a single paragraph to the LLM to be decomposed into propositions.

        Args:
            paragraph (str): The text paragraph to process.

        Returns:
            List[str]: A list of propositions extracted from the paragraph.
        """
        try:
            response = chat(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": f"Decompose the following:\n{paragraph}",
                    },
                ],
                model=self.model_name,
                format=Sentences.model_json_schema(),
            )
            sentences = Sentences.model_validate_json(response.message.content)
            return sentences.sentences
        except Exception as e:
            print(f"Error generating propositions for paragraph: {e}")
            return []

    def _modify_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """
        Merges small paragraphs together to create more contextually meaningful
        chunks before sending them to the LLM. This is useful when a document
        is split into too many small lines.

        Args:
            paragraphs (List[str]): A list of small text paragraphs.

        Returns:
            List[str]: A list of modified, longer paragraphs.
        """
        modified_paragraphs: List[str] = []
        current_paragraph: str = ""
        
        for para in paragraphs:
            if len(current_paragraph) < 100:
                current_paragraph += " " + para
            else:
                modified_paragraphs.append(current_paragraph.strip())
                current_paragraph = para
        
        if current_paragraph:
            modified_paragraphs.append(current_paragraph.strip())
            
        return modified_paragraphs

    def _process_text_concurrently(self, text: str) -> List[str]:
        """
        Splits text into paragraphs and processes them in parallel to extract
        propositions.

        Args:
            text (str): The full text content of a document.

        Returns:
            List[str]: A list of all propositions extracted from the text.
        """
        paragraphs = text.split("\n\n")
        
        # If the text is split into too many small paragraphs, merge them.
        if len(paragraphs) > 100:
            print("Too many paragraphs detected. Merging smaller paragraphs.")
            paragraphs = self._modify_paragraphs(paragraphs)
            
        all_propositions: List[str] = []
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_para = {
                executor.submit(self._get_propositions_from_paragraph, para): para
                for para in paragraphs
            }
            
            progress_bar = tqdm(
                as_completed(future_to_para),
                total=len(paragraphs),
                desc="Extracting Propositions",
                unit="para",
            )

            for future in progress_bar:
                try:
                    propositions_from_para = future.result()
                    if propositions_from_para:
                        all_propositions.extend(propositions_from_para)
                except Exception as e:
                    para = future_to_para[future]
                    print(f"Error processing paragraph '{para[:50]}...': {e}")
                    
        return all_propositions

    def extract(self) -> None:
        """
        The main method to iterate through all files, extract propositions,
        and then pass them to the AgenticChunker.
        """
        for filename, text in self.files.items():
            print(f"Extracting propositions from {filename}...")
            
            propositions = self._process_text_concurrently(text)
            
            if not propositions:
                print(f"No propositions were extracted from {filename}.")
                continue
                
            print(f"Extracted {len(propositions)} propositions from {filename}.")
            self.propositions[filename] = propositions
            
            # Once propositions are extracted for a file, chunk them immediately.
            agentic_chunker = AgenticChunker({filename: propositions})
            agentic_chunker.chunk_docs()
