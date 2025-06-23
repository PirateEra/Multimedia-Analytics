from flask import Flask, request, jsonify
import torch
from dotenv import load_dotenv
from src.utils.seed import seed_everything
from src.config import parse_args_llama
from src.model import load_model, llama_model_path
from src.dataset import load_dataset
from src.dataset.utils.retrieval import retrieval_via_pcst
from src.utils.lm_modeling import load_model as load_embed_model, load_text2embedding
from src.utils.collate import collate_fn
from graph_app_data import full_graph_data, all_temp_graph, temp_graph_data
from infer_sample import (
    multiple_queries,
    multiple_queries_unigram,
    multiple_queries_clauses,
    jaccard_similarity,
    format_edges_for_prompt,
    get_subgraph,
    predict_graph
)
import importlib
import logging
import subprocess
import sys
import nltk

# This checks if it's downloaded; only downloads if missing
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)



app = Flask(__name__)

# Load once on startup
ollama_process = subprocess.Popen(["ollama", "serve"])
args = parse_args_llama()
load_dotenv()
seed_everything(seed=args.seed)
query_placeholder = "placeholder"

# Load dataset module and models
module = importlib.import_module(f"src.dataset.{args.dataset}")
emb_model, emb_tokenizer, emb_device = load_embed_model[module.model_name]()
text2embedding = load_text2embedding[module.model_name]
whole_graph = full_graph_data(args)

@app.route("/infer", methods=["POST"])
def infer():
    # try:
    data = request.get_json()
    args.query = data["query"]
    args.sample_idx = data.get("sample_idx", 0)
    args.dataset = data["dataset"]

    module = importlib.import_module(f"src.dataset.{args.dataset}")

    sub_graph, sub_graph_desc = get_subgraph(
        args, module, emb_model, emb_tokenizer, emb_device, args.query, text2embedding
    )
    prompt, response = predict_graph(args, sub_graph_desc, args.query)
    sub_graph_info = temp_graph_data(sub_graph_desc)

    if args.jaccard:
        query_combinations = multiple_queries_clauses(args.query)
        jaccard_info = {}
        for word, temp_query in query_combinations.items():
            temp_graph, desc = get_subgraph(args, 
                                module, 
                                emb_model, 
                                emb_tokenizer, 
                                emb_device, 
                                temp_query, 
                                text2embedding)
            # Dictionairy of key being the missing word, and the value a tuple of the similarity value and subgraph
            similarity = jaccard_similarity(sub_graph, temp_graph)
            jaccard_info[word] = (1 - similarity, desc) # Invert the score by doing 1 - similarity
    else:
        jaccard_info = None # if we do not compute it, we return None

    whole_graph = full_graph_data(args)
    subgraphs = all_temp_graph(args, jaccard_info)

    return jsonify({
        "desc": sub_graph_info,
        "prompt": prompt,
        "response": response,
        "jaccard_info": jaccard_info,
        "whole_graph": whole_graph,
        "subgraphs": subgraphs
    }), 201

    # except Exception as e:
    #     return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)
