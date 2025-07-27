"""
Defines runtime configuration variables, primarily file system paths.

This module centralizes the paths used throughout the application for data
storage and retrieval, making it easy to manage the file structure from a
single location.
"""

# The path to the folder where source documents (e.g., PDFs) are stored.
DATA_FOLDER_PATH: str = "data"

# The path to the folder where persistent vector store data (ChromaDB) is saved.
VECTOR_DATA_PATH: str = "vector_data"
