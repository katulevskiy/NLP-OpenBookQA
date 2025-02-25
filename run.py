import numpy
import torch
import json

import os
import requests
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Retrieve the JWT token from the environment
JWT_TOKEN = os.getenv("JWT_TOKEN")

# Define the API endpoint
url = "http://localhost:3000/api/chat/completions"

# Prepare the payload
payload = {
    "model": "deepseek-r1:32b",
    "messages": [{"role": "user", "content": "Why is the sky blue?"}],
}

# Prepare the headers, using the JWT token from the environment
headers = {"Authorization": f"Bearer {JWT_TOKEN}", "Content-Type": "application/json"}

# Make the POST request
try:
    response = requests.post(url, json=payload, headers=headers)
    print("Status code:", response.status_code)
    print("Response:", response.json())
except requests.exceptions.RequestException as e:
    print("Request failed:", e)
except Exception as ex:
    print("Error decoding response:", ex)
