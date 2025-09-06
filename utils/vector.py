
import chromadb
import os
import logging
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import re
import time
import traceback
import shutil
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from concurrent.futures import ProcessPoolExecutor
import tempfile

logger = logging.getLogger(__name__)

CHROMA_DB_PATH="./sdg_chroma_db"

def track_performance(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def get_saved_vector_dbs(base_dir=CHROMA_DB_PATH):

    if not os.path.exists(base_dir):
        return []
    
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    all_collections = client.list_collections()

    collection_summaries = [
    (collection.name if not isinstance(collection.count(), Exception) else "Error retrieving count")
    for collection in all_collections
]

    saved_dbs=collection_summaries
    return saved_dbs

def save_vector_db(vector_db: Chroma, collection_name: str, base_dir: str = CHROMA_DB_PATH):

    # Validate input
    if vector_db is None:
        logger.error("Vector database is None")
        st.error("Cannot save an empty vector database")
        return None
   
    # Sanitize the hashed name to ensure it's file system friendly
    sanitized_collection_name = re.sub(r'[^\w\-_\. ]', '_', collection_name)
   
    # Create a full path for the specific collection
    collection_path = os.path.join(base_dir, sanitized_collection_name)
   
    # Ensure the base and collection directories exist
    os.makedirs(collection_path, exist_ok=True)
   
    # Comprehensive debug logging
    try:
        # Retrieve database contents
        db_contents = vector_db.get()
        
        # Debug logging of database contents
        if db_contents is None:
            logger.error("Database contents are None")
            st.error("No documents found in the vector database")
            return None
        
        # Extract database contents
        documents = db_contents.get('documents', [])
        embeddings = db_contents.get('embeddings', [])
        ids = db_contents.get('ids', [])
        
        # Log detailed debug information
        logger.info(f"Original collection name: {collection_name}")
        # logger.info(f"Hashed collection name: {hashed_collection_name}")
        logger.info(f"Saving to directory: {collection_path}")
        logger.info(f"Number of documents: {len(documents)}")
        logger.info(f"Number of embeddings: {len(embeddings)}")
        logger.info(f"Number of IDs: {len(ids)}")
        
        # Verify we have documents to save
        if not documents:
            logger.error("No documents to save in the vector database")
            st.error("Cannot save an empty vector database")
            return None
        
        # Attempt to persist the vector database
        try:
            # Use the embedding function from the original vector database
            Chroma.from_documents(
            documents=documents,
            embedding=vector_db._embedding_function,
            collection_name=collection_name,
            persist_directory=collection_path
        )
           
            logger.info(f"Vector database saved successfully")
            st.success(f"Vector database saved in '{collection_path}'")
            return True
        
        except Exception as persist_error:
            logger.error(f"Error persisting vector database: {persist_error}")
            st.error(f"Failed to persist vector database: {persist_error}")
            return None
    
    except Exception as e:
        logger.error(f"Error preparing vector database for saving: {e}")
        st.error(f"Failed to prepare vector database: {e}")
        return None
    
def list_and_diagnose_chroma_collections():

    try:
        # Use PersistentClient to access the database
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        
        # List all collections
        all_collections = client.list_collections()
        
        print("Available Collections:",all_collections)
        for collection in all_collections:
            print(f"- {collection.name}")
            try:
                # Try to get collection details
                count = collection.count()
                print(f"  Documents in collection: {count}")
            except Exception as count_error:
                print(f"  Error getting collection count: {count_error}")
        
        return all_collections
    
    except Exception as e:
        print(f"Error listing collections: {e}")
        traceback.print_exc()
        return []
    

def load_vector_db(collection_name: str):
    """
    Enhanced loading method with multiple fallback strategies.
    
    Args:
        collection_name (str): Name of the collection to load
    
    Returns:
        Chroma vector store or None if loading fails
    """
    # Verify chroma_db directory exists
    if not os.path.exists(CHROMA_DB_PATH):
        st.error("Chroma DB directory does not exist!")
        return None
    
    # First, try the alternative loading method
    vector_db = alternative_load_vector_db(collection_name)
    # print("Passed--alternative_load_vector_db")
    
    if vector_db is not None:
        return vector_db
    
    # If alternative method fails, try fallback strategies
    try:
        # List all collections in the directory
        collection_path = os.path.join(CHROMA_DB_PATH, collection_name)
        
        # Check if the specific collection directory exists
        if not os.path.exists(collection_path):
            st.error(f"Collection directory '{collection_name}' does not exist!")
            
            # List all available collection directories
            available_collections = [
                d for d in os.listdir(CHROMA_DB_PATH) 
                if os.path.isdir(os.path.join(CHROMA_DB_PATH, d))
            ]
            
            st.error("Available collections:")
            for collection in available_collections:
                st.error(f"- {collection}")
            
            return None
        
        # Use OllamaEmbeddings for consistency
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Initialize Chroma vector store with detailed logging
        vector_db = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=CHROMA_DB_PATH
        )
        
        # Attempt to retrieve contents
        db_contents = vector_db.get()
        
        # Print detailed information about the database
        print("Database Contents:")
        print(f"Number of IDs: {len(db_contents.get('ids', []))}")
        print(f"Number of Documents: {len(db_contents.get('documents', []))}")
        print(f"Number of Embeddings: {len(db_contents.get('embeddings', []))}")
        
        return vector_db
    
    except Exception as e:
        st.error(f"Error loading vector database: {e}")
        # traceback.print_exc()
        return None
    

def alternative_load_vector_db(collection_name: str):
    """
    Enhanced method to load Chroma vector database with comprehensive error handling.
   
    Args:
        collection_name (str): Name of the collection to load
   
    Returns:
        Chroma vector store or None if loading fails
    """
    # First, diagnose available collections
    available_collections = list_and_diagnose_chroma_collections()

    print("Inside available_collections", available_collections)
   
    # Check if the specific collection exists
    matching_collections = [
        collection for collection in available_collections
        if collection.name == collection_name
    ]
   
    if not matching_collections:
        st.error(f"Collection '{collection_name}' not found. Available collections are:")
        for collection in available_collections:
            st.error(f"- {collection.name}")
        return None
   
    try:
        # Use Chroma's PersistentClient directly
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
       
        # Try to get the collection
        collection = client.get_collection(name=collection_name)
       
        # Print collection details
        logger.info(f"Collection name: {collection.name}")
        logger.info(f"Number of items: {collection.count()}")
       
        # Use the same embedding model as when saving
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
       
        # Recreate Chroma vector store
        vector_db = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=CHROMA_DB_PATH
        )
       
        # Verify database contents
        db_contents = vector_db.get()
        
        if db_contents is None:
            logger.error("No contents found in the vector database")
            return None
        
        # Detailed logging with safe value checking
        ids = db_contents.get('ids', [])
        documents = db_contents.get('documents', [])
        embeddings_list = db_contents.get('embeddings', [])
        
        logger.info("Database Contents:")
        logger.info(f"Number of IDs: {len(ids)}")
        logger.info(f"Number of Documents: {len(documents)}")
        logger.info(f"Number of Embeddings: {len(embeddings_list) if embeddings_list is not None else 0}")
       
        # Additional verification
        if not ids or not documents:
            logger.warning("No documents found in the vector database")
            return None
        
        logger.info(f"Vector Database Object: {vector_db}")
       
        return vector_db
   
    except Exception as e:
        logger.error(f"Alternative loading method error: {e}")
        traceback.print_exc()
        return None
    
def get_sanitized_collection_name(filename: str) -> str:
    """
    Generate a sanitized collection name from a filename.
    
    Args:
        filename (str): Original filename
    
    Returns:
        str: Sanitized collection name
    """
    # Remove file extension and sanitize filename
    sanitized_name = "".join(
        c for c in filename.split('.')[0] 
        if c.isalnum() or c in ['-', '_']
    ).lower()
    return sanitized_name

def batch_embed(embeddings, texts, batch_size=32):
    """
    Manually implement batching for embeddings if needed
    """
    embedded_texts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embedded_batch = embeddings.embed_documents(batch)
        embedded_texts.extend(embedded_batch)
    return embedded_texts

def create_vector_db(file_upload, custom_name: str = None) -> Chroma:
    """
    Create a vector database from an uploaded PDF file.

    Args:
        file_upload (st.UploadedFile): Streamlit file upload object containing the PDF.

    Returns:
        Chroma: A vector store containing the processed document chunks.
    """
    logger.info(f"Creating vector DB from file upload: {file_upload.name}")

    # Use custom name or generate from filename with UUID for uniqueness
    if custom_name is None:
        sanitized_name = get_sanitized_collection_name(file_upload.name)
        collection_name = f"{sanitized_name}_rag_{uuid.uuid4().hex[:4]}"  # Add UUID for uniqueness
    else:
        collection_name = custom_name

    logger.info(f"Collection name used: {collection_name}")

    # Temporary directory for file processing
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, file_upload.name)

    with open(path, "wb") as f:
        f.write(file_upload.getvalue())
        logger.info(f"File saved to temporary path: {path}")

    # Efficient document loading
    loader = UnstructuredPDFLoader(path)
    data = loader.load()

    # Advanced text splitting with optimized parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Smaller, more precise chunks
        chunk_overlap=200,    # Improved context preservation
        length_function=len,
        is_separator_regex=False
    )

    # Parallel chunk processing
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        chunks = list(executor.map(text_splitter.split_documents, [data]))

    logger.info("Document split into chunks in parallel")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Option 1: Batch embed the document chunks before creating Chroma vector store
    document_texts = [chunk.page_content for chunk in chunks[0]]
    embedded_texts = batch_embed(embeddings, document_texts)


    # Create vector store with optimized settings
    vector_db = Chroma.from_documents(
        documents=chunks[0],  # Unpack from parallel processing
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=CHROMA_DB_PATH
    )

    logger.info(f"Vector DB created with collection name: {collection_name}")

    shutil.rmtree(temp_dir)
    logger.info(f"Temporary directory {temp_dir} removed")
    return vector_db