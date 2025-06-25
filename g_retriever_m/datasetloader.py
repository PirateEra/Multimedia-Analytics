import importlib
import torch
import pandas as pd
from src.utils.lm_modeling import load_model as load_embed_model, load_text2embedding
import os

class DatasetLoader:
    def __init__(self, dataset_name: str):
        self.dataset_module = importlib.import_module(f"src.dataset.{dataset_name}")
        
        model_name = getattr(self.dataset_module, 'model_name', None)
        
        self.emb_model, self.emb_tokenizer, self.emb_device = load_embed_model[model_name]()
        self.text2embedding = load_text2embedding[model_name]
        
        self.path_graphs = getattr(self.dataset_module, 'path_graphs')
        self.path_nodes = getattr(self.dataset_module, 'path_nodes')
        self.path_edges = getattr(self.dataset_module, 'path_edges')

    def load_graph_by_id(self, sample_idx):
        graph = torch.load(f"{self.path_graphs}/{sample_idx}.pt")
        nodes = pd.read_csv(f"{self.path_nodes}/{sample_idx}.csv")
        edges = pd.read_csv(f"{self.path_edges}/{sample_idx}.csv")
        return graph, nodes, edges

    def list_graph_ids(self):
        files = os.listdir(self.path_graphs)
        graph_ids = [os.path.splitext(f)[0] for f in files if f.endswith('.pt')]
        return sorted(graph_ids)
