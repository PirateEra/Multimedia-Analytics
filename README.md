# Multimedia analytics graph querying

This repository contains a demo application and supporting tools for retrieving and inferring facts from the WebQSP dataset using a graph-based approach.

## Environment setup

To set up the environment, use the provided `environment.yml` file. This ensures all dependencies are installed:

```bash
conda env create -f environment.yml
conda activate g_retriever_m
```

You can then run any of the scripts within the `g_retriever_m` folder.

### Ollama Setup (Required for LLM-based Retrieval)

The application relies on Ollama for LLM inference.

1. **Install Ollama:**
   Run the provided installation script:

```bash
bash scripts/get_ollama.sh
```

2. **Download the required LLM model:**

```bash
ollama pull llama3.2🥉b
```

This will download the correct LLM (Llama 3.2 3B) for use by the application.

### Get Hugging Face Access Token

1. Generate an access token: https://huggingface.co/docs/hub/en/security-tokens.
2. Add your token to the code file /g_retriever_m/.env as follows:
```bash
HF_TOKEN=YOUR_TOKEN
```

## Running the Demo App

To launch the interactive demo app, run:

```bash
python g_retriever_m/app.py
```

This starts the main application interface.

## Dataset Setup

To use the application, download the **WebQSP** dataset from the following link:

🔗 [Download WebQSP Dataset](https://drive.google.com/file/d/1REhbLnyeGKJ_NbaHQ4imuv20-0j5ZX6R/view?usp=sharing)

Then, place the dataset into the following folder structure:

```
g_retriever_m/dataset/webqsp/
```

Make sure the contents of the dataset are directly inside the `webqsp` folder.

## Manual Testing

You can test the retriever manually (without the interface) using `infer_sample.py`. Here's an example command:

```bash
python g_retriever_m/infer_sample.py --query "Give me an interesting fact about frank ocean" --dataset webqsp --sample_idx 0 --seed 1
```

## Scripts Overview

- `app.py`: Entry point for the demo app.
- `graph_app_data.py`: Contains helper functions to create graph data used by the app.
- `infer_sample.py`: Used for manual testing. Includes helper functions for inference.
- `api_utils.py`: Contains utility functions for easing inference in the app.
- `api_retriever.py`: A API interface for model inference, used for faster execution on systems like Snellius.

---
