''' Component for initializing and accessing embeddings and the large language model (LLM). '''
import logging

from app.config import BaseConfig as app_config
from app.utils.load_yaml_config import load_yaml_config
from app.ingest import initialize_sentence_transformer_embeddings
from app.llm import initialize_llm

logger = logging.getLogger(app_config.APP_NAME)

class SharedComponents:
    ''' Shared components for embeddings and LLM '''

    _embeddings_instance = None
    _llm_instance = None

    @classmethod
    def initialize_components(cls):
        """
        Initializes the embeddings and LLM instances.
        """
        if cls._embeddings_instance is None:
            embeddings_config = load_yaml_config("EMBEDDINGS_CONFIGURATION_PATH")

            if embeddings_config is None:
                raise ValueError("Failed to load embeddings configuration.")

            cls._embeddings_instance = initialize_sentence_transformer_embeddings(
                model_path=embeddings_config['embeddings_config']['model']
            )
            logging.info("Successfully loaded SentenceTransformer embeddings!")

        if cls._llm_instance is None:
            llm_config = load_yaml_config("MODEL_CONFIGURATION_PATH")

            if llm_config is None:
                raise ValueError("Failed to load LLM configuration.")

            cls._llm_instance = initialize_llm(llm_config)

    @classmethod
    def get_embeddings(cls):
        """
        Returns the embeddings instance.

        Returns:
            object: The embeddings instance for nlp.
        """
        cls.initialize_components()  # Initialize both embeddings and LLM
        return cls._embeddings_instance

    @classmethod
    def get_llm(cls):
        """
        Returns the language model (LLM) instance.

        Returns:
            object: The language model instance for nlp.
        """
        cls.initialize_components()  # Initialize both embeddings and LLM
        return cls._llm_instance
