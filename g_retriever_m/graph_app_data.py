import torch
import importlib
import networkx as nx
from torch_geometric.utils import to_networkx
import json
from networkx.readwrite import json_graph
import pandas as pd
from io import StringIO


def full_graph_data(args):
    dataset_module = importlib.import_module(f"src.dataset.{args.dataset}")
    id = args.sample_idx
    # read nodes and edges csv
    nodes = pd.read_csv(f"{dataset_module.path_nodes}/{id}.csv")
    edges = pd.read_csv(f"{dataset_module.path_edges}/{id}.csv")
    
    # create list of node data
    node_data = [{"data": {"id": row["node_id"], "label": row["node_attr"]}} for index, row in nodes.iterrows()]
    # dictionary that converts node id to node label
    node_data_dict = {node["data"]["id"]: node["data"]["label"] for node in node_data}
    # list of edge data
    edge_data = [{"data": {"source": node_data_dict[row["src"]], "target": node_data_dict[row["dst"]]}} for index, row in edges.iterrows()]
    #  edge_data = [{"data": {"source": node_data[row["src"]]["data"]["label"], "target": node_data[row["dst"]]["data"]["label"]}} for index, row in edges.iterrows()]
    # make full graph data
    graph_data = {f"Dataset {args.dataset}": {f"Graph {id}": {"nodes": node_data, "edges": edge_data}}}

    return graph_data


def temp_graph_data(subgr, id="main", list_subgr=None):
    # obtain the description of the subgraph
    if list_subgr:
        desc = list_subgr[subgr][1]
    else:
        desc = subgr
    # split the description into node desc and edge desc
    node_part, edge_part = desc.split("\n\n", 1)
    
    # read string as file
    node_string = StringIO(node_part)
    edge_string = StringIO(edge_part)
    
    # read string as csv
    nodes = pd.read_csv(node_string)
    edges = pd.read_csv(edge_string)
    
    # list of node data
    node_data = [{"data": {"id": row["node_id"], "label": row["node_attr"]}} for index, row in nodes.iterrows()]
    # dictionary of id:label
    node_data_dict = {node["data"]["id"]: node["data"]["label"] for node in node_data}
    # list of edge data
    edge_data = [{"data": {"source": node_data_dict[row["src"]], "target": node_data_dict[row["dst"]]}} for index, row in edges.iterrows()]
    # subgraph data
    graph_data = {f"Subgraph {id}": {"nodes": node_data, "edges": edge_data}}

    return graph_data

def all_temp_graph(args, list_subgr):
    # obtain list of subgraph data
    list_subgraphs = [temp_graph_data(subgr, id, list_subgr) for i, subgr in enumerate(list_subgr)]
    # make final dictionary
    final_graph_data = {f"Dataset {args.dataset}": list_subgraphs}
    
    return final_graph_data