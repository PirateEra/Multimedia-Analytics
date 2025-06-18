import os
import torch
# from src.utils.seed import seed_everything
from src.config import parse_args_llama
from src.model import load_model, llama_model_path
from src.dataset import load_dataset
from src.dataset.utils.retrieval import retrieval_via_pcst
from src.utils.lm_modeling import load_model as load_embed_model, load_text2embedding
import importlib
import pandas as pd
from src.utils.collate import collate_fn
# from torch_geometric.explain import Explanation
import networkx as nx
from torch_geometric.utils import to_networkx
import sys
import torch.nn as nn


def _load_model():
    sys.argv = ["",
                "--model_name", "graph_llm",
                "--llm_model_name", "7b_chat"]
    args = parse_args_llama()
    args.llm_model_path = llama_model_path[args.llm_model_name]

    model = load_model[args.model_name](
        args=args
    )
    model.eval()
    return model

def _load_embedders(dataset_model="sbert") -> dict:
    embedders = {}
    embedders["emb_model"], embedders["emb_tokenizer"], embedders["emb_device"] = load_embed_model[dataset_model]()
    embedders["text2embedding"] = load_text2embedding[dataset_model]
    return embedders

def _load_datasets() -> dict:
    datasets = {}
    for name in load_dataset.keys():
        datasets[name] = load_dataset[name]()
    return datasets

def process_query(query, dataset, model, graph_idx, embedders, dataset_name):
    dataset_module = importlib.import_module(f"src.dataset.{dataset_name}")
    # Determine which graph to use
    idx = graph_idx
    graph_id = idx
    # label = ""
    label = dataset[graph_id]["label"]

    graph = torch.load(f"{dataset_module.path_graphs}/{graph_id}.pt")
    nodes = pd.read_csv(f"{dataset_module.path_nodes}/{graph_id}.csv")
    edges = pd.read_csv(f"{dataset_module.path_edges}/{graph_id}.csv")

    # Encode user question and retrieve subgraph
    text2embedding = embedders["text2embedding"]
    q_emb = text2embedding(embedders["emb_model"], embedders["emb_tokenizer"], embedders["emb_device"], [query])[0]
    subg, desc, attn_nodes, attn_edges, mapping = retrieval_via_pcst(
        graph, q_emb, nodes, edges,
        topk=3, topk_e=5 if "webqsp" in dataset else 3, cost_e=0.5)

    # Compose sample for model inference
    question_prefix = "Question: "
    suffix = "\nAnswer:" if "webqsp" in dataset else "\n\nAnswer:"
    sample = {
        "id": idx,
        "question": f"{question_prefix}{query}{suffix}",
        "label": label,
        "graph": subg,
        "desc": desc,
    }
    sample_batch = collate_fn([sample])

    # Step 5: Inference
    with torch.no_grad():
        output = model.inference(sample_batch)

    # Step 6: Print result
    for i, pred in enumerate(output["pred"]):
        print(f"→ {pred}")

    return subg, pred, attn_nodes, attn_edges, dataset[0], mapping