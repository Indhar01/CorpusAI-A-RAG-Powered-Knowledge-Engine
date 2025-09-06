import streamlit as st
from utils.vector import (
    get_saved_vector_dbs,
    alternative_load_vector_db,
    get_sanitized_collection_name,
    save_vector_db,
    create_vector_db,
    load_vector_db,
)
from utils.general import extract_all_pages_as_images, process_question
from utils.logging_config import setup_logging
from utils.performance_utils import measure_time, PerformanceMonitor, profile_function

import logging
import os
import shutil
import chromadb
from functools import lru_cache
from utils.log_scheduler import start_cleanup_scheduler

# Add this line right after setting up the loggers
start_cleanup_scheduler()

# Set up logging
logger, perf_logger = setup_logging()

CHROMA_DB_PATH="./sdg_chroma_db"

# Page configuration
st.set_page_config(
    page_title="L1 SDG_Chat WEB",
    page_icon="😁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

@lru_cache(maxsize=32)
def get_chromadb_client(path: str) -> chromadb.PersistentClient:
    """Cached ChromaDB client initialization"""
    return chromadb.PersistentClient(path=path)

@measure_time
def load_collections(client: chromadb.PersistentClient):
    """Load collections with performance monitoring"""
    with PerformanceMonitor("collection_loading"):
        return client.list_collections()

@profile_function
def main() -> None:
    """
    Main function to run the Streamlit application with enhanced logging and performance monitoring.
    """
    logger.info("Starting application")
    st.subheader("✈️ L1 SDG_Chat")

    # Create layout
    col1, col2 = st.columns([1.5, 2])

    # Initialize session state with logging
    session_state_keys = {
        "messages": [],
        "vector_db": None,
        "pdf_pages": [],
        "use_sample": False,
    }
    
    for key, default in session_state_keys.items():
        if key not in st.session_state:
            st.session_state[key] = default
            logger.debug(f"Initialized session state key: {key}")

    # Vector DB Loading Section with performance monitoring
    with col1:
        st.markdown("### 📂 Saved Vector Databases")
        with PerformanceMonitor("vector_db_loading"):
            client = get_chromadb_client(CHROMA_DB_PATH)
            st.session_state["vector_db"] = client
            collections = load_collections(client)
            
        st.success("Loaded vector database:")
        with st.container(height=500, border=True):
            for i, collection in enumerate(collections, 1):
                st.write(f"{i}. {collection.name}")
                logger.debug(f"Displayed collection: {collection.name}")

    # Chat Interface with enhanced error handling
    with col2:
        st.markdown("### 💬 Chat Interface")
        message_container = st.container(height=500, border=True)
        
        # Apply custom CSS with logging
        try:
            message_container.markdown(
                """
                <style>
                .stContainer {
                    padding: 20px;
                    background-color: #f8f9fa;
                    border: 1px solid #dee2e6;
                    border-radius: 10px;
                    max-height: 600px;
                    overflow-y: auto;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
        except Exception as e:
            logger.error(f"Error applying custom CSS: {e}")

        # Display message history
        for i, message in enumerate(st.session_state["messages"]):
            avatar = "🤖" if message["role"] == "assistant" else "😎"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        # Handle user input with performance monitoring
        if prompt := st.chat_input("Enter a prompt here...", key="chat_input"):
            logger.info(f"Received user prompt: {prompt}")
            
            try:
                # Add user message
                st.session_state["messages"].append({"role": "user", "content": prompt})
                with message_container.chat_message("user", avatar="😎"):
                    st.markdown(prompt)

                # Process assistant response with performance monitoring
                with message_container.chat_message("assistant", avatar="🤖"):
                    with st.spinner(":green[processing...]"):
                        with PerformanceMonitor("process_question"):
                            if st.session_state.get("vector_db") is not None:
                                response = process_question(
                                    prompt, 
                                    st.session_state["vector_db"], 
                                   
                                )
                                st.markdown(response)
                                logger.info("Successfully processed question and generated response")
                            else:
                                st.warning("Please load or upload a vector database first.")
                                logger.warning("Attempted to process question without vector database")

                # Update chat history
                if st.session_state.get("vector_db") is not None:
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": response}
                    )

            except Exception as e:
                error_msg = f"Error processing prompt: {str(e)}"
                st.error(error_msg, icon="⛔️")
                logger.error(error_msg, exc_info=True)
        else:
            if st.session_state["vector_db"] is None:
                logger.warning("No vector database loaded")
                st.warning("Upload a PDF file or use the sample PDF to begin chat...")

if __name__ == "__main__":
    main()
