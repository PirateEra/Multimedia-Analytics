Certainly! Here's a concise intro + your existing run instructions:

---

# Graph Retriever App

This project provides an interactive interface for exploring and retrieving subgraphs from a knowledge graph based on natural language prompts.
It uses an LLM (via Ollama) to interpret prompts, extract entities, and visualize subgraph relevance.

---

## How to Run

1. **Set up the Conda environment:**

```bash
bash scripts/setup_conda_env.sh
```

2. **Install Ollama:**

```bash
bash scripts/get_ollama.sh
```

3. **Download the required Ollama model (e.g., Llama 3 8B):**

```bash
ollama pull llama3:8b
```

4. **Run the app:**

```bash
python g_retriever_m/app.py
```

---
