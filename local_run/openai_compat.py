"""Client for OpenAI-compatible chat endpoints (used here for the Ornith vLLM server).

Text only. Ornith advertises `attachment: false`, so it cannot take the render
images the refinement loop depends on -- visual feedback stays on Gemini.
"""

import os
import re
import time

import requests

RETRYABLE_STATUS = (408, 409, 429, 500, 502, 503, 504)

# Reasoning models on vLLM often emit their scratchpad inline.
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)

STOP_SEQUENCES = ["### [User message]", "<END>"]


def endpoint_config():
    """Read base URL and key for the compatible server from the environment."""

    base = os.getenv("ORNITH_BASE_URL") or os.getenv("OPENAI_COMPAT_BASE_URL")
    key = os.getenv("ORNITH_API_KEY") or os.getenv("OPENAI_COMPAT_API_KEY")
    if not base:
        raise ValueError("No ORNITH_BASE_URL set in local_run/.env")
    return base.rstrip("/"), key


def ask(model, prompt, temperature=0.2, max_tokens=32768, retries=4):
    """Send a single-turn chat completion and return the assistant's text."""

    base, key = endpoint_config()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stop": STOP_SEQUENCES,
    }

    last_error = None
    for attempt in range(retries):
        try:
            response = requests.post(f"{base}/chat/completions", headers=headers,
                                     json=payload, timeout=300)
        except requests.RequestException as exc:
            last_error = f"network error: {exc}"
            time.sleep(2 ** attempt)
            continue

        if response.status_code in RETRYABLE_STATUS:
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            wait = min(5 * (2 ** attempt), 60)
            print(f"  [{model}] {last_error} -- retrying in {wait}s")
            time.sleep(wait)
            continue

        if response.status_code != 200:
            raise RuntimeError(f"{model} error {response.status_code}: {response.text[:500]}")

        choice = response.json()["choices"][0]
        message = choice.get("message", {})
        text = (message.get("content") or "").strip()

        # Some servers split the scratchpad into its own field; others inline it.
        text = THINK_BLOCK.sub("", text).strip()

        if text:
            return text

        if choice.get("finish_reason") == "length":
            last_error = "response hit max_tokens before emitting an answer"
            payload["max_tokens"] = int(payload["max_tokens"] * 1.5)
            continue
        last_error = f"empty response (finish_reason={choice.get('finish_reason')})"
        time.sleep(2)

    raise RuntimeError(f"{model} failed after {retries} attempts. Last error: {last_error}")
