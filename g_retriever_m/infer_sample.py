# import os
# import torch
# from src.utils.seed import seed_everything
# from src.config import parse_args_llama
# from src.model import load_model, llama_model_path
# from src.dataset import load_dataset
# from src.utils.collate import collate_fn
# from dotenv import load_dotenv


# def main(args):
#     # Step 1: Reproducibility
#     seed_everything(seed=args.seed)
#     print(f"Loaded config for model {args.model_name} on dataset {args.dataset}")

#     # Step 2: Load dataset and model
#     dataset = load_dataset[args.dataset]()
#     args.llm_model_path = llama_model_path[args.llm_model_name]

#     model = load_model[args.model_name](
#         graph=dataset.graph,
#         graph_type=dataset.graph_type,
#         args=args
#     )
#     model.eval()

#     # Step 3: Get user question (from CLI args or stdin)
#     query = args.query

#     # Step 4: Prepare a sample
#     sample = dataset[0]  # use a sample entry
#     print(sample)
#     sample["question"] = query
#     sample_batch = collate_fn([sample])

#     # Step 5: Inference
#     with torch.no_grad():
#         output = model.inference(sample_batch)

#     # Step 6: Print result
#     for i, pred in enumerate(output["pred"]):
#         print(f"→ {pred}")


# if __name__ == "__main__":
#     args = parse_args_llama()
#     load_dotenv()
#     main(args)
#     torch.cuda.empty_cache()


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
# from torch_geometric.explain import Explanation
import networkx as nx
from torch_geometric.utils import to_networkx
# import matplotlib.pyplot as plt


def main(args):
    # Step 1: Reproducibility
    seed_everything(seed=args.seed)
    print(f"Loaded config for model {args.model_name} on dataset {args.dataset}")

    # Step 2: Load dataset and model
    dataset = load_dataset[args.dataset]()
    args.llm_model_path = llama_model_path[args.llm_model_name]

    model = load_model[args.model_name](
        # graph=dataset.graph,
        # graph_type=dataset.graph_type,
        args=args
    )
    model.eval()

    # Step 3: Get user question (from CLI args or stdin)
    # Step 3: Get user question (from CLI args)
    query = args.query

    # Step 4: Prepare a sample
    # sample = dataset[0]  # use a sample entry
    # print(sample)
    # sample["question"] = query
    # Load dataset module for access to path constants
    dataset_module = importlib.import_module(f"src.dataset.{args.dataset}")

    print('TEST')
    # Load embedding model used during preprocessing
    emb_model, emb_tokenizer, emb_device = load_embed_model["sbert"]()
    text2embedding = load_text2embedding["sbert"]

    # Determine which graph to use
    idx = args.sample_idx
    if hasattr(dataset, "questions"):
        # hij komt nooit hierin? is dit stukje nodig?
        row = dataset.questions.iloc[idx]
        graph_id = row.get("image_id", idx)
        label = row.get("answer", "")
    else:
        graph_id = idx
        # label = ""
        label = dataset[graph_id]["label"]

    graph = torch.load(f"{dataset_module.path_graphs}/{graph_id}.pt")
    nodes = pd.read_csv(f"{dataset_module.path_nodes}/{graph_id}.csv")
    edges = pd.read_csv(f"{dataset_module.path_edges}/{graph_id}.csv")

    # Encode user question and retrieve subgraph
    q_emb = text2embedding(emb_model, emb_tokenizer, emb_device, [query])[0]
    subg, desc, attn_nodes, attn_edges, mapping = retrieval_via_pcst(
        graph, q_emb, nodes, edges,
        topk=3, topk_e=5 if "webqsp" in args.dataset else 3, cost_e=0.5)

    # Compose sample for model inference
    question_prefix = "Question: "
    suffix = "\nAnswer:" if "webqsp" in args.dataset else "\n\nAnswer:"
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


if __name__ == "__main__":
    args = parse_args_llama()
    load_dotenv()
    subg, pred, attn_nodes, attn_edges, sample_0, mapping = main(args)
    print(subg)
    print(pred)
    print(attn_nodes)
    print(attn_edges)
    print(mapping)
    # Explanation(subg, edge_index=subg.edge_index, edge_attr=subg.edge_attr).visualize_graph(backend='networkx')
    # plt.show
    # subgraph_visualization = to_networkx(subg, to_undirected=True)
    subgraph_visualization = to_networkx(sample_0['graph'], to_undirected=True)
    # nx.draw(subgraph_visualization)
    # plt.savefig("test_subg_img1.png")
    torch.cuda.empty_cache()