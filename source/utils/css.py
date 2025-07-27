"""
Contains the CSS styles for the Streamlit user interface.

This module centralizes all custom CSS used to style the application,
making it easier to maintain and update the visual theme. The CSS is
stored in a single string constant that can be injected into the
Streamlit application.
"""

MAIN_CSS: str = """
<style>
    /* Import custom fonts from Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Barlow:wght@300;400;600&family=Barlow+Condensed:wght@300;400;600&display=swap');
    
    /* Apply base styles to the body */
    body {
        font-family: 'Barlow', sans-serif;
        background: linear-gradient(135deg, #ff9a9e, #fad0c4);
        color: #3D3D3D;
    }
    
    /* Style for individual chat messages */
    .stChatMessage {
        font-family: 'Barlow Condensed', sans-serif;
        padding: 10px;
        border-radius: 10px;
    }
    
    /* Style for messages from the Human/User */
    .stChatMessage.Human {
        background-color: rgba(239, 232, 220, 0.8);
        color: #3D3D3D;
        text-align: left;
    }
    
    /* Style for messages from the AI */
    .stChatMessage.AI {
        background-color: #ffcc00;
        color: #3D3D3D;
        text-align: right;
    }
    
    /* Style for the chat input box */
    .stChatInput input {
        font-family: 'Barlow', sans-serif;
        border: 2px solid #ff9a9e;
    }
</style>
"""
