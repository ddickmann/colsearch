
import os
import requests

api_key = os.getenv("OPENAI_API_KEY")
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
data = {
    "model": "gpt-4.1",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 5
}

try:
    response = requests.post("https://api.openai.com/v1/chat/completions", json=data, headers=headers)
    print(f"Status: {response.status_code}")
    print(response.text)
except Exception as e:
    print(e)
