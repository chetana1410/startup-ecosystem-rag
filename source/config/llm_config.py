"""
Defines configuration variables for the Language Models, embedding models,
and the RAG pipeline parameters.

This module centralizes all model names and tuning parameters, making it easy
to adjust the behavior of the core AI components from a single location.
"""

# --- Model Names ---
# The name of the primary Language Model to be used for generation tasks.
LLM_MODEL_NAME: str = "llama3.1:8b-instruct-fp16"

# The name of the SentenceTransformer model for creating text embeddings.
EMBEDDING_MODEL_NAME: str = "BAAI/bge-large-en-v1.5"

# The name of the CrossEncoder model for re-ranking retrieved documents.
RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-large"


# --- Chunking and Processing Parameters ---
# The maximum number of characters to include in a single agentic chunk.
CHUNK_SIZE_LIMIT: int = 1000

# The character limit for the truncated UUIDs used as chunk IDs.
CHUNK_ID_TRUNCATE_LIMIT: int = 7


# --- Retrieval and Re-ranking Parameters ---
# The default number of chunks to retrieve from the vector store in the initial pass.
NUMBER_OF_RETRIEVED_CHUNKS_LIMIT: int = 7

# The number of documents to retrieve from the summary-based vector store.
NUMBER_OF_RETRIEVED_DOCS_LIMIT: int = 25

# The number of top documents to keep after re-ranking the summary-retrieved docs.
NUMBER_OF_TOP_RERANK_SUMMARY_DOCS: int = 25

# The final number of top chunks to be passed to the generator after re-ranking.
NUMBER_OF_TOP_RERANK_CHUNKS: int = 3

# The score threshold to halt re-ranking. If the difference between two
# consecutive documents' scores is greater than this, the selection stops.
RERANK_SCORE_DIFFERECE_THRESHOLD: float = 0.3
