"""
This module provides a function for initializing the Large Language Model (LLM) using configurations
from the specified YAML file. The LLM is created using the LlamaCpp class from the langchain library.
It supports loading the model, setting up callback handlers, and handling configuration errors.
"""
import os
import logging
from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.llms import LlamaCpp
import yaml

from app.config import BaseConfig as app_config

logger = logging.getLogger(app_config.APP_NAME)

def initialize_llm() -> object:
    """
    Initialize Large Language Model (LLM) using configurations from YAML file.

    Returns:
        object: An instance of the LlamaCpp class initialized with the specified configurations.

    Raises:
        RuntimeError: If an error occurs during LLM initialization, a RuntimeError is raised
                      with a descriptive error message.

    """
    try:
        # Load up configurations
        load_dotenv()
        llm_config_path: str = os.environ.get("MODEL_CONFIGURATION_PATH")

        with open(llm_config_path, 'r', encoding="utf-8") as file:
            llm_config = yaml.safe_load(file)

        model_path: str = llm_config['llm_config']['model_path']

        logger.info('Trying to load language model on path %s', model_path)

        callback_manager = AsyncCallbackManager([StreamingStdOutCallbackHandler()])

        llm = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=llm_config['llm_config']['n_gpu_layers'],
            n_ctx=llm_config['llm_config']['n_ctx'],
            n_batch=llm_config['llm_config']['n_batch'],
            n_threads=llm_config['llm_config']['n_threads'],
            temperature=llm_config['llm_config']['temperature'],
            top_p=llm_config['llm_config']['top_p'],
            top_k=llm_config['llm_config']['top_k'],
            repeat_penalty=llm_config['llm_config']['repeat_penalty'],
            last_n_tokens_size=llm_config['llm_config']['last_n_tokens'],
            max_tokens=llm_config['llm_config']['max_tokens'],
            f16_kv=llm_config['llm_config']['f16_kv'],
            verbose=llm_config['llm_config']['verbose'],
            callback_manager=callback_manager,
        )

        logger.info("Successfully loaded model with pipeline!")
        return llm

    except Exception as e:
        logger.error(f"An error occurred during LLM initialization: {e}")
        raise RuntimeError("Failed to initialize Language Model (LLM). Please check configurations.") from e
