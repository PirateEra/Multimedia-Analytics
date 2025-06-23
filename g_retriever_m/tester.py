import requests
import json

API_URL = "http://145.136.62.2:5000/infer"

# Define the queries for each sample index
queries = {
    0: "Who is justin bieber's brother?",
    1: "Who is jonathan glickman?",
    2: "When was hurricane edith?"
}

results = {}

for sample_idx, query in queries.items():
    payload = {
        "dataset": "webqsp",
        "sample_idx": sample_idx,
        "query": query
    }

    response = requests.post(API_URL, json=payload)

    if response.status_code == 201:
        result = response.json()
        results[sample_idx] = {
            "query": query,
            "jaccard_info": result.get("jaccard_info"),

        }
        print(f"Sample {sample_idx} processed successfully.")
    else:
        print(f"Error for sample {sample_idx}: {response.status_code}")
        results[sample_idx] = {
            "query": query,
            "error": response.text
        }
