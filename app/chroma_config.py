''' Chroma config '''
import os
from dotenv import load_dotenv
import yaml

from chromadb.config import Settings

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

# Define document loaders mapping
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

# Load environment variables from .env file
load_dotenv()

# Get the path to the Chroma configuration file from the environment variables
chroma_config_path = os.environ.get("CHROMA_CONFIGURATION_PATH")

# Load Chroma configuration from YAML file
with open(chroma_config_path, 'r', encoding="utf-8") as file:
    chroma_config = yaml.safe_load(file)

# Create Chroma settings object
CHROMA_SETTINGS = Settings(
    persist_directory=chroma_config['chroma_config']['persist_directory_path'],
    chroma_db_impl=chroma_config['chroma_config']['chroma_db_impl'],
    anonymized_telemetry=chroma_config['chroma_config']['anonymized_telemetry'],
)
