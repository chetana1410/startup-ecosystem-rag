"""
This module defines the StreamlitApp class, which encapsulates the entire
user interface and application logic for the EcoAI chatbot.
"""

import sys
import time
from typing import Any, List, Tuple

import streamlit as st

from source.core.backend import Backend
from source.utils import MAIN_CSS

# Prevents Streamlit from scanning torch.classes, which can cause errors.
sys.modules["torch.classes"] = None


class StreamlitApp:
    """
    Manages the Streamlit front-end, including session state, UI components,
    and interaction with the backend.

    This class is responsible for:
    - Initializing and managing the session state for the application.
    - Rendering the chat interface, including user and AI messages.
    - Handling user input and orchestrating the call to the backend.
    - Displaying the response from the backend with a streaming effect.
    """

    def __init__(self, load_docs: bool) -> None:
        """
        Initializes the StreamlitApp.

        Args:
            load_docs (bool): A flag indicating whether to trigger the
                              document loading pipeline on backend initialization.
        """
        self.load_docs: bool = load_docs

    def _initialize_session_state(self) -> None:
        """
        Sets up the session state with default values if they don't exist.
        This ensures that the app's state is preserved across reruns.
        """
        defaults = {
            "messages": [],
            "query": "",
            "ai_response": "",
            "disabled_input": False,
            "waiting_for_response": False,
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

        # Initialize the backend only once and store it in the session state.
        if "backend" not in st.session_state:
            st.session_state["backend"] = Backend(self.load_docs)

    def _clear_input(self) -> None:
        """
        Callback function for the chat input submission.
        It moves the user input to the 'query' state and disables the input field.
        """
        st.session_state["query"] = st.session_state["user_input"]
        st.session_state["disabled_input"] = True

    def _stream_response(self) -> None:
        """
        A generator function that yields words from the AI response with a
        short delay to create a streaming effect in the UI.
        """
        for word in st.session_state.get("ai_response", "").split(" "):
            yield word + " "
            time.sleep(0.02)

    def _get_response(self, query: str) -> Tuple[str, float, str]:
        """
        Calls the backend to get a response for the user's query.

        Args:
            query (str): The user's question.

        Returns:
            Tuple[str, float, str]: A tuple containing the AI's response,
                                    the confidence score, and the sources.
        """
        backend: Backend = st.session_state["backend"]
        response, confidence, sources = backend.run(query)
        return response, confidence, sources

    def _display_messages(self) -> None:
        """
        Renders the chat history from the session state to the UI.
        """
        for sender, content, sources, confidence, stream in st.session_state["messages"]:
            if sender == "user":
                st.chat_message("Human").write(content, unsafe_allow_html=True)
            else:
                ai_message = f"{content}\n\n**Source:** {sources}\n**Confidence:** {confidence:.2%}"
                if stream:
                    st.session_state["ai_response"] = ai_message
                    st.chat_message("AI", avatar="logo.jpeg").write_stream(
                        self._stream_response
                    )
                    # Mark the message as "not to be streamed" after displaying
                    st.session_state["messages"][-1] = (sender, content, sources, confidence, False)
                else:
                    st.chat_message("AI", avatar="logo.jpeg").write(
                        ai_message, unsafe_allow_html=True
                    )

    def run(self) -> None:
        """
        The main method to run the Streamlit application.
        It sets up the page, initializes state, and handles the main app loop.
        """
        st.set_page_config(page_title="EcoAI", page_icon="logo.jpeg")
        st.markdown(MAIN_CSS, unsafe_allow_html=True)

        self._initialize_session_state()

        # Header with logo and title
        col1, col2 = st.columns([1, 5])
        with col1:
            st.image("logo.jpeg", width=100)
        with col2:
            st.markdown("# EcoAI")

        self._display_messages()

        # Handle the query submission and response generation logic
        if st.session_state.get("waiting_for_response"):
            with st.status("Searching for relevant context in Database...."):
                response, confidence, sources = self._get_response(st.session_state["query"])
            
            st.session_state["messages"].append(("ai", response, sources, confidence, True))
            st.session_state["waiting_for_response"] = False
            st.session_state["disabled_input"] = False
            st.session_state["query"] = ""
            st.rerun()

        if st.session_state.get("query") and not st.session_state.get("waiting_for_response"):
            st.session_state["messages"].append(("user", st.session_state["query"], "", 0.0, False))
            st.session_state["waiting_for_response"] = True
            st.rerun()

        st.chat_input(
            placeholder="Ask your question...",
            key="user_input",
            disabled=st.session_state["disabled_input"],
            on_submit=self._clear_input,
        )

