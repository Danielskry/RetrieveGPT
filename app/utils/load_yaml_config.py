import os
import yaml
from dotenv import load_dotenv

def load_yaml_config(file_path):
    """
    Load configuration from the specified YAML file.

    Args:
        file_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration loaded from the YAML file.
    """
    try:
        # Check if the path is provided
        if not file_path and not isinstance(file_path, str):
            raise ValueError("File path is not provided.")

        # Load environment variables from .env file
        load_dotenv()

        config_path : str = os.environ.get(file_path)

        # Load configuration from YAML file
        with open(config_path, 'r', encoding="utf-8") as file:
            config = yaml.safe_load(file)

        return config
    except FileNotFoundError as fnfe:
        print(f"File not found: {fnfe}")
        return None
    except yaml.YAMLError as yamle:
        print(f"YAML error: {yamle}")
        return None
