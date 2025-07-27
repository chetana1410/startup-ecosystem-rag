"""
The `config` package centralizes all configuration variables for the application.

This __init__.py file makes the 'config' directory a Python package and exposes
key constants from its submodules. This allows for easy and consistent access
to configuration values like model names, file paths, and processing parameters
from anywhere in the application.
"""

from typing import List

from .llm_config import (
    CHUNK_ID_TRUNCATE_LIMIT,
    CHUNK_SIZE_LIMIT,
    EMBEDDING_MODEL_NAME,
    LLM_MODEL_NAME,
    NUMBER_OF_RETRIEVED_CHUNKS_LIMIT,
    NUMBER_OF_RETRIEVED_DOCS_LIMIT,
    NUMBER_OF_TOP_RERANK_CHUNKS,
    NUMBER_OF_TOP_RERANK_SUMMARY_DOCS,
    RERANKER_MODEL_NAME,
    RERANK_SCORE_DIFFERECE_THRESHOLD,
)
from .run_config import DATA_FOLDER_PATH, VECTOR_DATA_PATH

# Defines the public API for the 'config' package.
# When a client uses 'from source.config import *', only these names will be imported.
__all__: List[str] = [
    # from llm_config.py
    "LLM_MODEL_NAME",
    "EMBEDDING_MODEL_NAME",
    "RERANKER_MODEL_NAME",
    "CHUNK_SIZE_LIMIT",
    "CHUNK_ID_TRUNCATE_LIMIT",
    "NUMBER_OF_RETRIEVED_CHUNKS_LIMIT",
    "NUMBER_OF_RETRIEVED_DOCS_LIMIT",
    "NUMBER_OF_TOP_RERANK_CHUNKS",
    "NUMBER_OF_TOP_RERANK_SUMMARY_DOCS",
    "RERANK_SCORE_DIFFERECE_THRESHOLD",
    # from run_config.py
    "DATA_FOLDER_PATH",
    "VECTOR_DATA_PATH",
]
