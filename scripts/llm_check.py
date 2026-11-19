"""Make one direct OpenRouter chat-completions request and report diagnostics."""

import os
from pathlib import Path

import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=True)

model = os.getenv("OPENROUTER_MODEL", "")
api_key = os.getenv("OPENROUTER_API_KEY", "")
print(f"Model: {model or '(not set)'}")
print(f"API key set: {bool(api_key)}")

try:
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}" if api_key else "",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Reply with the single word OK"}],
        },
        timeout=30,
    )
    print(f"HTTP status: {response.status_code}")
    print(f"Response body: {response.text}")
except requests.RequestException as exc:
    print("HTTP status: unavailable")
    print(f"Response body: {type(exc).__name__}: {exc}")
