"""
This includes methods for initializing and accessing embeddings and the language model (LLM).
"""

from ingest import initialize_sentence_transformer_embeddings
from llm import initialize_llm

class SharedComponents:
    def __init__(self):
        """
        Initializes a SharedComponents.
        """
        self.embeddings_instance = initialize_sentence_transformer_embeddings(
            model_name="models/all-MiniLM-L6-v2"
            )
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
