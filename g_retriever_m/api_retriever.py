from flask import Flask, request, jsonify
import json
from api_utils import _load_model, _load_datasets, _load_embedders, process_query
from src.utils.seed import seed_everything

# Create Flask app instance
app = Flask('retriever')

# Application-wide state
model = None
datasets = None
embedders = None

# Function to initialize heavy components with app context
def initialize_app():
    global model, datasets, embedders

    with app.app_context():
        seed_everything(seed=1)
        model = _load_model()
        datasets = _load_datasets()
        embedders = _load_embedders()
        print("Model and datasets loaded successfully.")


# Define a route that uses shared resources
@app.route('/process_query', methods=['POST'])
def handle_process_query():
    dataset_name = request.args.get('dataset')
    graph_idx = int(request.args.get('graph'))
    query = request.args.get('query')

    if dataset_name not in datasets:
        return jsonify({"error": f"Unknown dataset: {dataset_name}"}), 400

    dataset = datasets[dataset_name]
    result = process_query(query, dataset, model, graph_idx, embedders, dataset_name)
    return jsonify(result)


# Main entry point
if __name__ == '__main__':
    initialize_app()
    app.run(host="0.0.0.0", port=5000, debug=False, load_dotenv=False)
