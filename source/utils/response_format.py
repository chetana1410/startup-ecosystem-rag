"""
Defines the Pydantic models for structuring and validating the output 
from the Language Model.

These models ensure that the LLM's responses adhere to a predefined format,
making the output reliable and easier to parse, especially when using
features like function calling or structured output generation.
"""

from typing import List, Optional
from pydantic import BaseModel

class Sentences(BaseModel):
    """
    A Pydantic model to validate a list of sentences.
    
    This is typically used to structure the output when an LLM is asked to 
    generate propositions or break down text into individual statements.

    Attributes:
        sentences (List[str]): A list where each item is a string representing a sentence.
    """
    sentences: List[str]

class ChunkID(BaseModel):
    """
    A Pydantic model to validate a chunk identifier.
    
    The chunk ID can be optional, allowing for cases where it might not be 
    present or applicable in an LLM's response.
    
    Attributes:
        chunk_id (Optional[str]): A string representing the unique identifier 
                                  of a text chunk, or None.
    """
    chunk_id: Optional[str]

class Summary(BaseModel):
    """
    A Pydantic model to validate a summary text.

    This ensures the LLM provides a summary in the expected string format.
    
    Attributes:
        summary (str): A string containing the summary of a document or text.
    """
    summary: str

class Title(BaseModel):
    """
    A Pydantic model to validate a document title.

    Used to ensure the title is extracted or generated as a simple string.
    
    Attributes:
        title (str): A string representing the title.
    """
    title: str
