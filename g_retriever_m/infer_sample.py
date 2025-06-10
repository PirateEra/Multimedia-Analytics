import os
import torch
from src.utils.seed import seed_everything
from src.config import parse_args_llama
from src.model import load_model, llama_model_path
from src.dataset import load_dataset
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
    query = args.query

    # Step 4: Prepare a sample
    sample = dataset[0]  # use a sample entry
    print(sample)
    sample["question"] = query
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
