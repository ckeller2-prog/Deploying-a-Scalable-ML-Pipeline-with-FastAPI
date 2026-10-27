import requests

url = "http://127.0.0.1:8003"

# Test GET request
response_get = requests.get(url)
print(f"Status Code: {response_get.status_code}")
print(f"Result: {response_get.json().get('greeting')}\n")

# Test POST request with sample data
payload = {
    "age": 39,
    "workclass": "State-gov",
    "fnlwgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

response_post = requests.post(f"{url}/predict/", json=payload)
print(f"Status Code: {response_post.status_code}")
print(f"Result: {response_post.json().get('result')}")
