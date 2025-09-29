import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_MODELS_URL = "https://models.github.ai/inference/chat/completions"
GITHUB_API_VERSION = "2022-11-28"

def llm_explain(role: str, context: str, query: str) -> str:
    if not GITHUB_TOKEN:
        return f"[LLM unavailable] Role: {role}\n{context}"
    try:
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
            "Content-Type": "application/json"
        }
        payload = {
            "model": "openai/gpt-4.1-mini",
            "messages": [
                {"role": "system", "content": f"You are {role}. Provide concise logistics insights."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuery:\n{query}"},
            ],
            "temperature": 0.1
        }
        resp = requests.post(GITHUB_MODELS_URL, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[LLM error] {e}\n\nContext:\n{context}"
