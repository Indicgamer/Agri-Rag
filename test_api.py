"""
Quick test script to check OpenRouter API status
"""
import urllib.request
import json

# Read .env manually
import os
env_file = os.path.join(os.path.dirname(__file__), ".env")
api_key = None
model = "minimax/minimax-m2.5:free"

if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            if line.startswith("OPENROUTER_API_KEY="):
                api_key = line.strip().split("=")[1]
            elif line.startswith("OPENROUTER_MODEL="):
                model = line.strip().split("=")[1]

print(f"Testing model: {model}")
print(f"API key present: {'Yes' if api_key else 'No'}")

# Simple test
test_prompt = "Say hello in one word."

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/anomalyco/opencode",
    "X-Title": "Agri-RAG"
}

data = json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": test_prompt}],
    "max_tokens": 20
}).encode()

print("Making API call...")
try:
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=15) as response:
        result = response.read()
        print(f"✅ SUCCESS! Response: {result.decode()[:200]}")
        
except urllib.error.HTTPError as e:
    if e.code == 429:
        print(f"❌ RATE LIMITED: {e.read().decode()[:200]}")
    else:
        print(f"⚠️ Error {e.code}: {e.read().decode()[:200]}")
        
except Exception as e:
    print(f"❌ Error: {e}")