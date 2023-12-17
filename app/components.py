''' Methods for initializing and accessing embeddings and the large language model (LLM). '''
import logging

from app.config import BaseConfig as app_config
from app.utils.load_yaml_config import load_yaml_config
from app.ingest import initialize_sentence_transformer_embeddings
from app.llm import initialize_llm

logger = logging.getLogger(app_config.APP_NAME)

class SharedComponents:
    ''' Shared components for embeddings and LLM '''

    def __init__(self):
        """
        Initializes a SharedComponents.
        """

        embeddings_config = load_yaml_config("EMBEDDINGS_CONFIGURATION_PATH")

        if embeddings_config is None:
            raise ValueError("Failed to load embeddings configuration.")

        self.embeddings_instance = initialize_sentence_transformer_embeddings(
            model_path=embeddings_config['embeddings_config']['model']
        )
        logging.info("Successfully loaded SentenceTransformer embeddings!")
        self.llm_instance = initialize_llm()

    def initialize_llm(self):
        """
        Initializes the language model (LLM) instance.

        """
        return self.llm_instance

    def get_embeddings(self):
        """
        Returns the embeddings instance.

        Returns:
            object: The embeddings instance for nlp.

        """
        return self.embeddings_instance

    def get_llm(self):
        """
        Returns the language model (LLM) instance.

        Returns:
            object: The language model instance for nlp.

        """
        return self.llm_instance
