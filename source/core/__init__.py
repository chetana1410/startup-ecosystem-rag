"""
The `core` package contains the central business logic and primary classes
that drive the EcoAI application.

This __init__.py file makes the 'core' directory a Python package and exposes
its key components, making them easily accessible from other parts of the
application, such as the main `app.py` entry point. By defining `__all__`,
we provide a clear public API for the package.
"""

from typing import List

from .agentic_chunker import AgenticChunker
from .backend import Backend
from .doc_loader import DocLoader
from .embeddings import Embeddings
from .generator import Generator
from .propositions import ExtractPropositions
from .retriever import Retriever
from .streamlit import StreamlitApp
from .vectordb import VectorDB

# Defines the public API for the 'core' package.
# When a client uses 'from source.core import *', only these names will be imported.
__all__: List[str] = [
    "AgenticChunker",
    "Backend",
    "DocLoader",
    "Embeddings",
    "ExtractPropositions",
    "Generator",
    "Retriever",
    "StreamlitApp",
    "VectorDB",
]
