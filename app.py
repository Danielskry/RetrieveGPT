#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from flask import Flask
from flask_cors import CORS
from components import SharedComponents

from ingest import ingest_data
from retrieve import get_answer

app = Flask(__name__)
CORS(app)

# Initialize the LLM
shared_components = SharedComponents()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/debug.log"),
        logging.StreamHandler()
    ]
)

@app.route('/ingest', methods=['GET'])
def route_ingest_data():
    return ingest_data(shared_components)

@app.route('/get_answer', methods=['POST'])
def route_get_answer():
    return get_answer(shared_components)

if __name__ == '__main__':
    # Run the app
    app.run(host="0.0.0.0", debug=True)
