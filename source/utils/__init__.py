"""
The `utils` package contains various helper modules and utilities for the EcoAI application.

This __init__.py file makes the 'utils' directory a Python package and exposes
key components from its submodules, making them easily accessible from other
parts of the application. By defining `__all__`, we provide a clear public API
for the package.
"""

from typing import List

from .css import MAIN_CSS
from .prompts import (
    DOC_SUMMARY_SYSTEM_PROMPT,
    FIND_RELEVANT_CHUNK_SYSTEM_PROMPT,
    GENERATOR_SYSTEM_PROMPT,
    NEW_CHUNK_SUMMARY_SYSTEM_PROMPT,
    NEW_CHUNK_TITLE_SYSTEM_PROMPT,
    PROPOSITIONS_SYSTEM_PROMPT,
    UPDATE_CHUNK_SUMMARY_SYSTEM_PROMPT,
    UPDATE_CHUNK_TITLE_SYSTEM_PROMPT,
)
from .response_format import ChunkID, Sentences, Summary, Title

# Defines the public API for the 'utils' package.
# When a client uses 'from source.utils import *', only these names will be imported.
__all__: List[str] = [
    # from css.py
    "MAIN_CSS",
    # from prompts.py
    "PROPOSITIONS_SYSTEM_PROMPT",
    "UPDATE_CHUNK_SUMMARY_SYSTEM_PROMPT",
    "UPDATE_CHUNK_TITLE_SYSTEM_PROMPT",
    "NEW_CHUNK_SUMMARY_SYSTEM_PROMPT",
    "NEW_CHUNK_TITLE_SYSTEM_PROMPT",
    "FIND_RELEVANT_CHUNK_SYSTEM_PROMPT",
    "DOC_SUMMARY_SYSTEM_PROMPT",
    "GENERATOR_SYSTEM_PROMPT",
    # from response_format.py
    "Sentences",
    "ChunkID",
    "Summary",
    "Title",
]
