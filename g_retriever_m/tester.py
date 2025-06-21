import requests

# URL of your Flask API
API_URL = "http://145.136.62.2:5000/infer"

# JSON body with inference parameters
payload = {
    "dataset": "webqsp",                     # The dataset to use
    "sample_idx": 0,                         # The graph/sample index
    "query": "Who is justin bieber brother?"      # Your natural language query
}

# Send the POST request with JSON data
response = requests.post(API_URL, json=payload)

# Handle the response
if response.status_code == 201:
    print("Success!")
    result = response.json()
    print("Prediction:", result["response"])
    print("\nPrompt:", result["prompt"])
    # print("\nJaccard Info:", result.get("jaccard_info"))
else:
    print("Error", response.status_code)
    print(response.text)
