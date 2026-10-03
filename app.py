"""
Main entry point for the EcoAI Streamlit application.

This script initializes and runs the Streamlit user interface.
"""

import os
import warnings
from source.core import StreamlitApp

def main() -> None:
    """
    Initializes and runs the Streamlit application.
    
    Suppresses DeprecationWarning for a cleaner console output and
    launches the main application loop.
    """
    # Suppress DeprecationWarning which is common with Streamlit and its dependencies.
    warnings.simplefilter("ignore", category=DeprecationWarning)
    
    # Set LOAD_DOCS=1 on the first run to build the local index from data/.
    load_docs = os.getenv("LOAD_DOCS", "0") == "1"
    app = StreamlitApp(load_docs=load_docs)
    
    # Start the Streamlit application server and render the UI.
    app.run()

if __name__ == "__main__":
    main()
