"""
This module is designed for ingesting data (documents in various formats such as pdf, txt, etc.) into our Chroma Vector Database.

It provides functions for loading documents from a specified directory, initializing SentenceTransformer embeddings, and ingesting 
data into Chroma Vector Database.
"""
import os
import glob
import logging
from typing import List, Any
from chromadb.config import Settings
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from flask import jsonify

from langchain.vectorstores.chroma import Chroma

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
        persist_directory="db/",
        chroma_db_impl='duckdb+parquet',
        anonymized_telemetry=False
)

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

# Define a mapping of file extensions to custom document loaders and their arguments
DOC_LOADERS_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}

class CustomSentenceTransformerEmbeddings:
    """
    A custom wrapper for the SentenceTransformer library, providing methods to embed
    documents and queries using pre-trained sentence embeddings.

    Attributes:
        embedding_function (SentenceTransformer): The SentenceTransformer model used for embeddings.

    Methods:
        __init__(self, model_name):
            Initializes the CustomSentenceTransformerEmbeddings instance with a specified
            SentenceTransformer model by name.

        embed_documents(self, texts):
            Embeds a list of documents into numerical vectors using the underlying
            SentenceTransformer model.

            Args:
                texts (List[str]): A list of text documents.

            Returns:
                List[List[float]]: A list of embedded vectors, each represented as a list of floats.

        embed_query(self, text):
            Embeds a single query text into a numerical vector using the underlying
            SentenceTransformer model.

            Args:
                text (str): The query text to be embedded.

            Returns:
                List[float]: The embedded vector represented as a list of floats.
    """

    def __init__(self, model_name):
        """
        Initializes the CustomSentenceTransformerEmbeddings instance with a specified
        SentenceTransformer model by name.

        Args:
            model_name (str): The name of the SentenceTransformer model to be used for embeddings.
        """
        self.embedding_function = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        """
        Embeds a list of documents into numerical vectors using the underlying
        SentenceTransformer model.

        Args:
            texts (List[str]): A list of text documents.

        Returns:
            List[List[float]]: A list of embedded vectors, each represented as a list of floats.
        """
        embeddings = self.embedding_function.encode(texts, convert_to_numpy=True).tolist()
        return [list(map(float, e)) for e in embeddings]

    def embed_query(self, text):
        """
        Embeds a single query text into a numerical vector using the underlying
        SentenceTransformer model.

        Args:
            text (str): The query text to be embedded.

        Returns:
            List[float]: The embedded vector represented as a list of floats.
        """
        embeddings = self.embedding_function.encode([text], convert_to_numpy=True).tolist()
        return [list(map(float, e)) for e in embeddings][0]

def initialize_sentence_transformer_embeddings(model_name: str) -> object:
    """
    Initialize SentenceTransformer embeddings.

    Parameters:
    - model_name (str): The name of the pre-trained SentenceTransformer model.

    Returns:
    - object: An object with methods for embedding documents and queries.

    Raises:
    - RuntimeError: If an error occurs during embedding initialization.
    """

    try:
        logging.info("Loading SentenceTransformer embedding function...")
        return CustomSentenceTransformerEmbeddings(model_name)

    except Exception as exception:
        raise RuntimeError(f"Error initializing custom embeddings: {exception}") from exception

def load_single_document(file_path: str) -> Document:
    """
    Load a document based on its file extension.

    Parameters:
    - file_path (str): The path to the document file.

    Returns:
    - Document: The loaded document.

    Raises:
    - ValueError: If the file extension is not supported.
    """

    try:
        ext = "." + file_path.rsplit(".", 1)[-1]
        if ext in DOC_LOADERS_MAPPING:
            loader_class, loader_args = DOC_LOADERS_MAPPING[ext]
            loader = loader_class(file_path, **loader_args)

            logging.info("Document loaded successfully!")
            return loader.load()[0]

        raise ValueError(f"Unsupported file extension '{ext}'")

    except Exception as exception:
        raise ValueError(f"Error loading document: {exception}") from exception

def load_documents_from_directory(source_dir: str) -> List[Document]:
    """
    Load multiple documents from a source directory using various loaders.

    Parameters:
    - source_dir (str): The directory containing the documents.

    Returns:
    - List[Document]: A list of loaded documents.

    Raises:
    - RuntimeError: If an error occurs during document loading.
    """

    try:
        all_files = []
        for ext in DOC_LOADERS_MAPPING:
            all_files.extend(
                glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
            )
        return [load_single_document(file_path) for file_path in all_files]

    except Exception as exception:
        raise RuntimeError(f"Error loading documents from directory: {exception}") from exception

def ingest_data(shared_components: object) -> Any:
    """
    Ingest data from source documents into Chroma Vector DB.

    Parameters:
    - shared_components (object): Shared components object containing necessary dependencies.

    Returns:
    - Any: The response indicating the success of the ingestion.

    Raises:
    - RuntimeError: If an error occurs during the ingestion process.
    """

    try:
        # Load environment variables
        persist_directory = "db"  # Chroma Vector DB
        source_directory = "source_documents"  # Where documents are loaded from

        # Load documents and split into chunks
        logging.info('Loading documents from %s', source_directory)

        chunk_size = 500
        chunk_overlap = 50
        documents = load_documents_from_directory(source_directory)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            )
        texts = text_splitter.split_documents(documents)

        logging.info('Loaded %s documents from %s', len(documents), source_directory)
        logging.info('Split into %s chunks of text (max. %s characters each)', len(texts), chunk_size)

        # Create embeddings
        embeddings = shared_components.get_embeddings()

        # Ingest into Chroma Vector DB
        chroma_db = Chroma.from_documents(
            texts,
            embeddings,
            persist_directory=persist_directory,
            client_settings=CHROMA_SETTINGS,
            )
        chroma_db.persist()
        chroma_db = None

        logging.info("Successfully ingested data!")
        return jsonify(response="Successfully ingested data!")

    except Exception as exception:
        raise RuntimeError(f"Error during data ingestion: {exception}") from exception
