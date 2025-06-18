import requests

# URL of your Flask API
API_URL = "http://145.136.62.59:5000/process_query"

# Parameters for the query
params = {
    "dataset": "webqsp",              # Change this to your dataset name
    "graph": 0,                       # Change this to the graph index
    "query": "Where was Einstein born?"  # Your natural language query
}

# Send the POST request
response = requests.post(API_URL, params=params)

# Handle the response
if response.status_code == 200:
    print("Success!")
    result = response.json()
    print(result)
else:
    print("Error", response.status_code)
    print(response.text)
