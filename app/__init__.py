''' Initializes Flask application '''
import logging.config
from os import environ

from dotenv import load_dotenv
from flask import Flask
from flask_cors import CORS

from .config import config as app_config

def create_app() -> Flask:
    ''' Create Flask app '''

    # Load environment
    load_dotenv()

    APPLICATION_ENV : str = get_environment()
    logging.config.dictConfig(app_config[APPLICATION_ENV].LOGGING)

    app = Flask(app_config[APPLICATION_ENV].APP_NAME)
    app.config.from_object(app_config[APPLICATION_ENV])

    CORS(app, resources={r'/api/*': {'origins': '*'}})

    app.config.from_prefixed_env()

    return app

def get_environment():
    ''' Get environment '''
    return environ.get('APPLICATION_ENV') or 'development'
