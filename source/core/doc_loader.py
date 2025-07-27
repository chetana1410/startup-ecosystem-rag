"""
This module defines the DocLoader class, responsible for loading and parsing
text content from PDF documents in a specified folder.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from source.config import VECTOR_DATA_PATH
from tqdm import tqdm
from unstructured.partition.pdf import partition_pdf


class DocLoader:
    """
    Loads and processes PDF files from a folder, extracting structured text.

    This class handles the entire document ingestion pipeline:
    1. Identifies PDF files in a given directory.
    2. Filters out files that have already been processed to support incremental updates.
    3. Uses the `unstructured` library to partition PDFs into text elements.
    4. Processes these elements to reconstruct clean, readable text.
    5. Leverages multi-threading to process multiple documents in parallel for efficiency.

    Attributes:
        folder_path (str): The path to the folder containing the PDF files.
        file_to_id_dict (Dict[str, str]): A mapping from filenames to unique IDs.
    """

    def __init__(self, folder_path: str) -> None:
        """
        Initializes the DocLoader.

        Args:
            folder_path (str): The path to the directory containing PDFs.

        Raises:
            FileNotFoundError: If the 'id_to_file_map.json' file is not found.
        """
        self.folder_path: str = folder_path
        self.file_to_id_dict: Dict[str, str] = {}
        
        try:
            with open("./id_to_file_map.json", "r") as f:
                id_to_file_map: Dict[str, str] = json.load(f)
            # Invert the map for quick lookup from filename to ID
            self.file_to_id_dict = {v: k for k, v in id_to_file_map.items()}
        except FileNotFoundError:
            print("Error: 'id_to_file_map.json' not found. Please ensure the file exists.")
            # Depending on requirements, you might want to raise the error
            # raise

    def _process_file(self, filename: str) -> Dict[str, str]:
        """
        Processes a single PDF file to extract its text content.

        Args:
            filename (str): The name of the PDF file to process.

        Returns:
            Dict[str, str]: A dictionary mapping the filename to its extracted text.
        """
        file_path = os.path.join(self.folder_path, filename)
        
        # Use unstructured to partition the PDF into elements
        elements = partition_pdf(
            file_path, infer_table_structure=True, strategy="hi_res"
        )

        text_parts: List[str] = []
        current_paragraph: str = ""
        
        for element in elements:
            # Ignore common non-content elements
            if element.category in ["Header", "Footer", "Image", "Table"]:
                continue
            
            # Stop processing if a 'References' section is found
            if element.category == "Title" and element.text.lower().startswith("references"):
                break
            
            # For narrative text, append it and start a new paragraph
            if element.category == "NarrativeText":
                current_paragraph += element.text
                text_parts.append(current_paragraph)
                current_paragraph = ""
            else:
                # For other text types, append with a newline
                current_paragraph += element.text + "\n"

            # To avoid overly long paragraphs, split if a certain length is reached
            if len(current_paragraph) > 300:
                text_parts.append(current_paragraph)
                current_paragraph = ""
        
        # Add any remaining text
        if current_paragraph:
            text_parts.append(current_paragraph)
            
        full_text = "\n\n".join(text_parts)
        return {filename: full_text}

    def load_docs(self) -> Dict[str, str]:
        """
        Loads all unprocessed PDF documents from the folder path.

        It identifies all PDFs, filters out those already present in the
        vector data path, and processes the remaining ones in parallel.

        Returns:
            Dict[str, str]: A dictionary mapping filenames to their extracted text.
        """
        print("Extracting Data from PDFs...")

        # Get a list of all PDF files in the specified directory
        try:
            all_files = [f for f in os.listdir(self.folder_path) if f.endswith(".pdf")]
        except FileNotFoundError:
            print(f"Error: The directory '{self.folder_path}' was not found.")
            return {}

        # Filter out files that have already been processed and vectorized
        files_to_process = [
            f
            for f in all_files
            if not os.path.isdir(
                os.path.join(VECTOR_DATA_PATH, f"Collection_{self.file_to_id_dict.get(f)}")
            )
        ]

        if not files_to_process:
            print("No new documents to process.")
            return {}

        all_extracted_text: Dict[str, str] = {}

        # Use a thread pool to process files concurrently for performance
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            # Create a future for each file processing task
            future_to_filename = {
                executor.submit(self._process_file, filename): filename
                for filename in files_to_process
            }
            
            # Process futures as they complete
            progress_bar = tqdm(
                as_completed(future_to_filename),
                total=len(files_to_process),
                desc="Processing PDFs",
                unit="file",
            )
            
            for future in progress_bar:
                filename = future_to_filename[future]
                try:
                    extracted_data = future.result()
                    all_extracted_text.update(extracted_data)
                    print(f"Successfully processed: {filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

        return all_extracted_text
