import os
import torch
from src.utils.seed import seed_everything
from src.config import parse_args_llama
from src.model import load_model, llama_model_path
from src.dataset import load_dataset
from src.dataset.utils.retrieval import retrieval_via_pcst
from src.utils.lm_modeling import load_model as load_embed_model, load_text2embedding
import importlib
import pandas as pd
from src.utils.collate import collate_fn
from dotenv import load_dotenv
import networkx as nx
from torch_geometric.utils import to_networkx
from pprint import pprint
import ollama
from itertools import combinations
from graph_app_data import full_graph_data, all_temp_graph

def multiple_queries(query):
    words = query.split()
    result = {}
    for i in range(len(words)):
        removed_word = words[i]
        remaining = words[:i] + words[i+1:]
        result[removed_word] = " ".join(remaining)

    return result


def jaccard_similarity(graph_1, graph_2):
    # Get needed information
    graph_1 = (to_networkx(graph_1, node_attrs= ['x'], edge_attrs = ['edge_attr'], to_undirected=True)).edges()
    graph_2 = (to_networkx(graph_2, node_attrs= ['x'], edge_attrs = ['edge_attr'], to_undirected=True)).edges()

    # Actual jaccard computation
    i = set(graph_1).intersection(graph_2)
    jacc_sim = len(i) / (len(graph_1) + len(graph_2) - len(i))

    return round(jacc_sim, 3)

def format_edges_for_prompt(desc):
    # Split the node section and edges section
    parts = desc.strip().split('\n\n')
    if len(parts) != 2:
        return "Error: Description format is not correct."
    node_section, edge_section = parts

    # Parse node lines
    node_lines = node_section.strip().split('\n')[1:]  # skip header
    nodes = {}
    unknown_count = 1

    for line in node_lines:
        # split the node and its label/attribute it is often of the case 1,label
        node_id, node_info = line.split(',', 1)
        node_id = node_id.strip()
        node_info = node_info.strip().strip('"')

        # Extract "name" if it exists (occurs in the scene_graphs dataset) example: 2,"name: desk; (x,y,w,h): (0, 482, 400, 51)"
        if "name:" in node_info:
            start = node_info.find(":") + 1
            end = node_info.find(";", start)
            label = node_info[start:end].strip()

        # Turn it into a unkown entity if the label starts with m. (happens in the webqsp dataset)
        elif (node_info.startswith("m.")):
            label = f"UnknownEntity_{unknown_count}"
            unknown_count += 1
        else:
            label = node_info

        nodes[node_id] = label

    # Turn the edges into a format where its not nodeid -> edge but label -> edge
    edge_lines = edge_section.strip().split('\n')[1:]  # skip header
    edge_strings = []
    for line in edge_lines:
        source_node, relation, destination_node = line.strip().split(',')
        source_label = nodes.get(source_node.strip(), "Unknown")
        destination_label = nodes.get(destination_node.strip(), "Unknown")
        edge_strings.append(f"{source_label} → {relation.strip()} → {destination_label}")

    return "\n".join(edge_strings)

def get_subgraph(args, module, emb_model, emb_tokenizer, emb_device, query, text2embedding):
    # Load the actual needed data to get a subgraph (so this loads the entire graph)
    id = args.sample_idx
    graph = torch.load(f"{module.path_graphs}/{id}.pt")
    nodes = pd.read_csv(f"{module.path_nodes}/{id}.csv")
    edges = pd.read_csv(f"{module.path_edges}/{id}.csv")
    
    # Encode user question and retrieve subgraph
    q_emb = text2embedding(emb_model, emb_tokenizer, emb_device, [query])[0]
    sub_graph, desc = retrieval_via_pcst(
        graph, q_emb, nodes, edges,
        topk=3, topk_e=5 if "webqsp" in args.dataset else 3, cost_e=0.5)

    return sub_graph, desc


def predict_graph(args, desc, query):
    # Create the actual question for the LLM (this was prompt engineered through trial and error)
    question = f"Based on the above graph only, and only use the above graph info. Answer the following question: {query}\
    \nKeep in mind UnknownEntity_* represents unknown labels of nodes in the graph, you may infer relationships through them.\
    \nif you think that the graph does not provide the right or enough info to answer, then mention that instead.\
    \nStart with your definite answer and after a very short brief explanation on how you got the answer."

    formatted_edges = format_edges_for_prompt(desc)
    prompt = f"{formatted_edges}\n\n{question}"

    response = ollama.chat(
        model='llama3.2:3b',
        messages=[
            {'role': 'system', 'content': 'You are a precise reasoning assistant that answers questions based only on the provided graph information.'},
            {'role': 'user', 'content': prompt}
        ],
        options={
        'seed': args.seed
        }
    )
    return prompt, response['message']['content']

def infer_sample(args):
    # Step 1: Reproducibility
    seed_everything(seed=args.seed)

    # Step 2: Get user question from the given command line args
    query = args.query

    # Load the dataset module for the used dataset (to get nodes and edges based on the structure of that datasets)
    dataset_module = importlib.import_module(f"src.dataset.{args.dataset}")

    # Load embedding model used during preprocessing
    emb_model, emb_tokenizer, emb_device = load_embed_model[dataset_module.model_name]()
    text2embedding = load_text2embedding[dataset_module.model_name]

    # Retrieve the subgraph and its description
    sub_graph, desc = get_subgraph(args, 
                                   dataset_module, 
                                   emb_model, 
                                   emb_tokenizer, 
                                   emb_device, 
                                   query, 
                                   text2embedding)

    # generate the prediction with an LLM
    prompt, response = predict_graph(args, desc, query)
    print(f"\n=== Graph Description ===\n{desc}\n====================\n")

    print(f"\n=== Model Prompt ===\n{prompt}\n====================\n")
    print(f"\n=== Model Answer ===\n{response}\n====================\n")

    # if the jaccard similarity should be computed (default True)
    if args.jaccard:
        query_combinations = multiple_queries(query)
        jaccard_info = {}
        for word, temp_query in query_combinations.items():
            temp_graph, desc = get_subgraph(args, 
                                   dataset_module, 
                                   emb_model, 
                                   emb_tokenizer, 
                                   emb_device, 
                                   temp_query, 
                                   text2embedding)
            # Dictionairy of key being the missing word, and the value a tuple of the similarity value and subgraph
            similarity = jaccard_similarity(sub_graph, temp_graph)
            jaccard_info[word] = (similarity, desc)
    else:
        jaccard_info = None # if we do not compute it, we return None

    return sub_graph, response, prompt, jaccard_info


if __name__ == "__main__":
    args = parse_args_llama()
    load_dotenv()
    subg, pred, prompt, jaccard_info = infer_sample(args)
    print(jaccard_info)
    whole_graph = full_graph_data(args)
    # print(whole_graph)
    subgraphs = all_temp_graph(args, jaccard_info)
    # print(subgraphs)
    torch.cuda.empty_cache()