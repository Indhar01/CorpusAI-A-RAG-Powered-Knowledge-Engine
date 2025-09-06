import logging
import pdfplumber
from langchain_community.chat_models import ChatOllama
# from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, List, Any, Tuple
from .query import process_query
import chromadb
import os
# from dotenv import load_dotenv
# load_dotenv()

logger = logging.getLogger(__name__)

CHROMA_DB_PATH="./sdg_chroma_db"

def extract_model_names(_models_info: Dict[str, List[Dict[str, Any]]],) -> Tuple[str, ...]:

    logger.info("Extracting model names from models_info")
    
    model_names = tuple(model["name"] for model in _models_info["models"])
    print("Model:",model_names)
    logger.info(f"Extracted model names: {model_names}")
    return model_names

def extract_all_pages_as_images(file_upload) -> List[Any]:

    logger.info(f"Extracting all pages as images from file: {file_upload.name}")
    pdf_pages = []
    with pdfplumber.open(file_upload) as pdf:
        pdf_pages = [page.to_image().original for page in pdf.pages]
    logger.info("PDF pages extracted as images")
    return pdf_pages

def optimized_retrieval(vector_db: Chroma, question: str, top_k: int = 5):
    """
    Advanced document retrieval with Maximal Marginal Relevance
    """
    
    logger.info(f"Retrieving relevant documents for question: {question}")
    
    retriever = vector_db.as_retriever(
        search_type="mmr",  # Maximal Marginal Relevance
        search_kwargs={
            "k": top_k,          # Number of top documents
            "fetch_k": 20,       # Initial fetch before filtering
            "lambda_mult": 0.5   # Balance between diversity and relevance
        }
    )
    
    relevant_docs = retriever.get_relevant_documents(question)
    logger.info(f"Retrieved {len(relevant_docs)} relevant documents")
    
    return relevant_docs


def process_question(question: str, vector_db: Chroma) -> str:

    logger.info(f"Processing question: {question}")

    try:

        client=chromadb.PersistentClient(
            path=CHROMA_DB_PATH
        )

        # query_handler = MultiCollectionQuery(client)

        collection_names = client.list_collections()
        

        llm = ChatOllama(model="llama3", temperature=0)

        # llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)


        logger.info(f"Loading Retriever..")

        # context_docs = optimized_retrieval(vector_db, question)
        
        logger.info(f"Loaded QA Chain")

        results = process_query(
            client=client,
            collections=collection_names,
            query_text=question,
            n_results=5,
            distance_threshold=2.0
        )
        
        # print("Result from Query Handler: ",results)
        

        # Compact prompt engineering
        template =  """
        You are an expert assistant specializing in support team ticket handling. Using only the most relevant information from the provided context, deliver a concise, accurate answer. Be sure to include the page number, section title, and URL path (if available) for reference.

        Context: {context}

        Question: {question}

        Answer (include page number, section title, and URL, if applicable):

        """

        prompt = PromptTemplate.from_template(template)

        # Streamlined chain
        chain = (
            prompt 
            | llm 
            | StrOutputParser()
        )

        # Prepare context with collection names and documents
        context_lines = []
        for result in results:
            collection_name = result['collection']
            for document in result['documents']:
                context_lines.append(f"Collection: {collection_name} | Document: {document}")
        
        # Combine the context lines into a single string
        response_context = "\n".join(context_lines)

        response = chain.invoke({
            "context": response_context,
            "question": question
        })

        return response
    
    except Exception as e:
        print(f"Error: {str(e)}")
        raise