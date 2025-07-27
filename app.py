"""
Main entry point for the EcoAI Streamlit application.

This script initializes and runs the Streamlit user interface.
"""

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
    
    # Initialize the main application class.
    # `load_docs=False` likely defers document loading until a user action,
    # improving initial startup time.
    app = StreamlitApp(load_docs=False)
    
    # Start the Streamlit application server and render the UI.
    app.run()

if __name__ == "__main__":
    main()
