"""Google AI Studio (Gemini) client for text and image prompts.

Deliberately built on `requests` only, so the local setup needs no SDK beyond
what a plain Python install already has.
"""

import base64
import json
import mimetypes
import os
import time

import requests

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Short names for the CLI; anything else is sent to the API verbatim.
ALIASES = {
    "gemini": "gemini-2.5-flash",
    "flash": "gemini-2.5-flash",
    "pro": "gemini-2.5-pro",
}

# The prompts in src/prompts.py are few-shot transcripts, so the model must be
# stopped before it invents another turn of the conversation.
STOP_SEQUENCES = ["### [User message]", "<END>"]

RETRYABLE_STATUS = (429, 500, 502, 503, 504)


def resolve_model(name):
    return ALIASES.get(name, name)


def _call(model, api_key, parts, temperature, max_tokens, retries, response_json):
    if not api_key:
        raise ValueError("No Gemini API key. Set GEMINI_API_KEY in local_run/.env")

    config = {
        "temperature": float(temperature),
        "maxOutputTokens": int(max_tokens),
    }
    if response_json:
        config["responseMimeType"] = "application/json"
    else:
        config["stopSequences"] = STOP_SEQUENCES

    budget = os.getenv("GEMINI_THINKING_BUDGET")
    if budget:
        config["thinkingConfig"] = {"thinkingBudget": int(budget)}

    payload = {"contents": [{"role": "user", "parts": parts}], "generationConfig": config}
    url = API_URL.format(model=resolve_model(model))
    headers = {"x-goog-api-key": api_key, "content-type": "application/json"}

    last_error = None
    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=180)
        except requests.RequestException as exc:
            last_error = f"network error: {exc}"
            time.sleep(2 ** attempt)
            continue

        # The free tier returns 429 regularly when queries run back to back.
        if response.status_code in RETRYABLE_STATUS:
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            wait = min(5 * (2 ** attempt), 60)
            print(f"  [gemini] {last_error} -- retrying in {wait}s")
            time.sleep(wait)
            continue

        if response.status_code != 200:
            raise RuntimeError(f"Gemini error {response.status_code}: {response.text[:500]}")

        data = response.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError(
                f"Gemini returned no candidates (promptFeedback={data.get('promptFeedback', {})})"
            )

        candidate = candidates[0]
        text = "".join(
            part["text"]
            for part in candidate.get("content", {}).get("parts", [])
            if "text" in part and not part.get("thought")
        ).strip()

        if text:
            return text

        reason = candidate.get("finishReason")
        if reason == "MAX_TOKENS":
            # Thinking tokens ate the budget; a larger allowance usually fixes it.
            last_error = "hit maxOutputTokens before emitting text"
            max_tokens = int(max_tokens * 1.5)
            payload["generationConfig"]["maxOutputTokens"] = max_tokens
            continue
        raise RuntimeError(f"Gemini returned empty text (finishReason={reason})")

    raise RuntimeError(f"Gemini failed after {retries} attempts. Last error: {last_error}")


def ask(model, api_key, prompt, temperature=0.2, max_tokens=16384, retries=4):
    """Send a text prompt and return the model's reply."""

    return _call(model, api_key, [{"text": prompt}], temperature, max_tokens, retries, False)


def ask_with_image(model, api_key, prompt, image_path, temperature=0.2,
                   max_tokens=16384, retries=4, want_json=False):
    """Send a prompt plus a rendered image and return the model's reply."""

    mime = mimetypes.guess_type(image_path)[0] or "image/png"
    with open(image_path, "rb") as fh:
        encoded = base64.b64encode(fh.read()).decode("ascii")

    parts = [{"text": prompt}, {"inline_data": {"mime_type": mime, "data": encoded}}]
    return _call(model, api_key, parts, temperature, max_tokens, retries, want_json)


def ask_json(model, api_key, prompt, image_path=None, temperature=0.0, retries=4):
    """Ask for a JSON object and parse it, tolerating stray prose or fences."""

    if image_path:
        raw = ask_with_image(model, api_key, prompt, image_path, temperature,
                             retries=retries, want_json=True)
    else:
        raw = _call(model, api_key, [{"text": prompt}], temperature, 16384, retries, True)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            return json.loads(raw[start:end + 1])
        raise
