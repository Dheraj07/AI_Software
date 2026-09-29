import requests
import time

url = "http://localhost:11434/api/chat"

payload = {
    "model": "qwen2.5-coder:14b",
    "messages": [
        {
            "role": "user",
            "content": "Reply with exactly HELLO"
        }
    ],
    "stream": False
}

print("Sending request to Ollama...")

start = time.time()

response = requests.post(
    url,
    json=payload,
    timeout=180
)

print(f"Response received in {time.time() - start:.2f} seconds")

print("Status:", response.status_code)

print("Response:")

print(response.json())