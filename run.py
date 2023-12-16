''' Flask app setup '''
from flask import jsonify
from app import create_app

from app.components import SharedComponents

from app.ingest import ingest_data
from app.retrieve import get_answer

app = create_app()

# Initialize the shared components for LLM and embeddings
shared_components = SharedComponents()

@app.route('/status', methods=['GET'])
def status():
    ''' App status endpoint '''
    return jsonify({'message': 'Running!'})

@app.route('/ingest', methods=['GET'])
def route_ingest_data():
    ''' Ingest data endpoint '''
    return ingest_data(shared_components)

@app.route('/get_answer', methods=['POST'])
def route_get_answer():
    ''' Get answer endpoint '''
    return get_answer(shared_components)

if __name__ == '__main__':
    app.run()
