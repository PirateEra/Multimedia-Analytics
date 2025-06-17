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


def main(args):
    # Step 1: Reproducibility
    seed_everything(seed=args.seed)
    print(f"Loaded config for model {args.model_name} on dataset {args.dataset}")

    # Step 2: Load dataset and model
    dataset = load_dataset[args.dataset]()
    args.llm_model_path = llama_model_path[args.llm_model_name]

    model = load_model[args.model_name](
        graph=dataset.graph,
        graph_type=dataset.graph_type,
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

    # Load embedding model used during preprocessing
    emb_model, emb_tokenizer, emb_device = load_embed_model[dataset_module.model_name]()
    text2embedding = load_text2embedding[dataset_module.model_name]

    # Determine which graph to use
    idx = args.sample_idx
    if hasattr(dataset, "questions"):
        row = dataset.questions.iloc[idx]
        graph_id = row.get("image_id", idx)
        label = row.get("answer", "")
    else:
        graph_id = idx
        label = ""

    graph = torch.load(f"{dataset_module.path_graphs}/{graph_id}.pt")
    nodes = pd.read_csv(f"{dataset_module.path_nodes}/{graph_id}.csv")
    edges = pd.read_csv(f"{dataset_module.path_edges}/{graph_id}.csv")

    # Encode user question and retrieve subgraph
    q_emb = text2embedding(emb_model, emb_tokenizer, emb_device, [query])[0]
    subg, desc = retrieval_via_pcst(
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


if __name__ == "__main__":
    args = parse_args_llama()
    load_dotenv()
    main(args)
    torch.cuda.empty_cache()