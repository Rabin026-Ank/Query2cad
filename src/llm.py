import os
import re
import time
import base64
import mimetypes
import json
import requests

# ---------------------------------------------------------------------------
# Backends
#
#   gemini / gemini-*  -> Google AI Studio          (text + images)
#   ornith, or any other unrecognised name -> the OpenAI-compatible server in
#                                             .env                (text + images)
#   codellama, llama3  -> Together AI               (text, as in the paper)
#   chatgpt, gpt4-turbo-> OpenAI                    (text, as in the paper)
#
# The together and openai SDKs are imported lazily so the system runs with only
# `requests` installed when those backends are not used.
# ---------------------------------------------------------------------------

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

GEMINI_ALIASES = {
    "gemini": "gemini-2.5-flash",
    "gemini-flash": "gemini-2.5-flash",
    "gemini-pro": "gemini-2.5-pro",
}

TOGETHER_MODELS = {"codellama", "llama3"}
OPENAI_MODELS = {"chatgpt", "gpt4-turbo"}

# The prompts in prompts.py are few-shot transcripts, so generation has to stop
# before the model invents another turn of the conversation.
STOP_SEQUENCES = ["### [User message]", "<END>"]

RETRYABLE_STATUS = (408, 409, 429, 500, 502, 503, 504)

# Reasoning models served through vLLM often emit their scratchpad inline.
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)


def is_gemini(model):
    return model.startswith("gemini") or model in GEMINI_ALIASES


def is_compatible_endpoint(model):
    """True for models served by the OpenAI-compatible endpoint (e.g. ornith)."""

    return not (is_gemini(model) or model in TOGETHER_MODELS or model in OPENAI_MODELS)


def _encode_image(image_path):
    mime = mimetypes.guess_type(image_path)[0] or "image/png"
    with open(image_path, "rb") as fh:
        return mime, base64.b64encode(fh.read()).decode("ascii")


# ---------------------------------------------------------------------------
# Google AI Studio
# ---------------------------------------------------------------------------

def _gemini_call(model, api_key, parts, temp, max_tokens=16384, retries=4, want_json=False):
    if not api_key:
        raise ValueError("No Gemini API key. Set GEMINI_API_KEY in .env")

    config = {"temperature": float(temp), "maxOutputTokens": int(max_tokens)}
    if want_json:
        config["responseMimeType"] = "application/json"
    else:
        config["stopSequences"] = STOP_SEQUENCES

    budget = os.getenv("GEMINI_THINKING_BUDGET")
    if budget:
        config["thinkingConfig"] = {"thinkingBudget": int(budget)}

    payload = {"contents": [{"role": "user", "parts": parts}], "generationConfig": config}
    url = GEMINI_URL.format(model=GEMINI_ALIASES.get(model, model))
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
            print(f"  [{model}] {last_error} -- retrying in {wait}s")
            time.sleep(wait)
            continue

        if response.status_code != 200:
            raise RuntimeError(f"Gemini error {response.status_code}: {response.text[:500]}")

        data = response.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError(
                f"Gemini returned no candidates (promptFeedback={data.get('promptFeedback', {})})")

        candidate = candidates[0]
        text = "".join(
            part["text"]
            for part in candidate.get("content", {}).get("parts", [])
            if "text" in part and not part.get("thought")
        ).strip()

        if text:
            return text

        if candidate.get("finishReason") == "MAX_TOKENS":
            # Thinking tokens ate the budget; a larger allowance usually fixes it.
            last_error = "hit maxOutputTokens before emitting text"
            max_tokens = int(max_tokens * 1.5)
            payload["generationConfig"]["maxOutputTokens"] = max_tokens
            continue
        raise RuntimeError(f"Gemini returned empty text (finishReason={candidate.get('finishReason')})")

    raise RuntimeError(f"Gemini failed after {retries} attempts. Last error: {last_error}")


# ---------------------------------------------------------------------------
# OpenAI-compatible endpoint (Ornith)
# ---------------------------------------------------------------------------

def _endpoint_config():
    base = os.getenv("ORNITH_BASE_URL") or os.getenv("OPENAI_COMPAT_BASE_URL")
    key = os.getenv("ORNITH_API_KEY") or os.getenv("OPENAI_COMPAT_API_KEY")
    if not base:
        raise ValueError("No ORNITH_BASE_URL set in .env")
    return base.rstrip("/"), key


def _compatible_call(model, content, temp, max_tokens=32768, retries=4):
    base, key = _endpoint_config()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": float(temp),
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
        text = THINK_BLOCK.sub("", (choice.get("message", {}).get("content") or "")).strip()

        if text:
            return text

        if choice.get("finish_reason") == "length":
            last_error = "hit max_tokens before emitting an answer"
            payload["max_tokens"] = int(payload["max_tokens"] * 1.5)
            continue
        last_error = f"empty response (finish_reason={choice.get('finish_reason')})"
        time.sleep(2)

    raise RuntimeError(f"{model} failed after {retries} attempts. Last error: {last_error}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_answers(model, api_key, query, temp, base_url=None) -> str:
    """
    Given the prompt: Get the LLM response for that particular prompt.
    """

    if is_gemini(model):
        output = _gemini_call(model, api_key, [{"text": query}], temp)
        print(output)

        return output

    elif model == "codellama":
        os.environ["TOGETHER_API_KEY"] = api_key
        url = 'https://api.together.xyz/inference'
        headers = {
        'Authorization': 'Bearer ' + os.environ["TOGETHER_API_KEY"],
        'accept': 'application/json',
        'content-type': 'application/json'
        }
        data = {
        "model": "codellama/CodeLlama-70b-hf",
        "prompt": query,
        "max_tokens": 3000,
        "temperature": temp,
        "top_p": 0.7,
        "top_k": 50,
        "repetition_penalty": 1,
        "stop": ["<|EOT|>","<|begin▁of▁sentence|>","<|end▁of▁sentence|>", "<|\s|>", "### [User message]", "\n\n\n\n\n"]
        }
        response = requests.post(url, json=data, headers=headers)
        text = response.json()['output']['choices'][0]['text']
        print(text)

        return text

    elif model == "chatgpt":
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)
        message = {
            'role' : 'user',
            'content' : query
        }
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[message],
            stop=["<END>", "### [User message]", "\n\n\n\n\n"],
            temperature=temp,
            max_tokens=3000
        )
        output = response.choices[0].message.content
        print(output)

        return output

    elif model == 'gpt4-turbo':
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)
        message = {
            'role' : 'user',
            'content' : query
        }
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[message],
            stop=["<END>", "### [User message]", "\n\n\n\n\n"],
            temperature=temp,
            max_tokens=3000
        )
        output = response.choices[0].message.content
        print(output)

        return output

    elif model == "llama3":
        os.environ["TOGETHER_API_KEY"] = api_key
        url = 'https://api.together.xyz/inference'
        headers = {
        'Authorization': 'Bearer ' + os.environ["TOGETHER_API_KEY"],
        'accept': 'application/json',
        'content-type': 'application/json'
        }
        data = {
        "model": "meta-llama/Llama-3-70b-chat-hf",
        "prompt": query,
        "max_tokens": 3000,
        "temperature": temp,
        "top_p": 0.7,
        "top_k": 50,
        "repetition_penalty": 1,
        "stop": ["<|EOT|>","<|begin▁of▁sentence|>","<|end▁of▁sentence|>", "<|\s|>", "### [User message]", "\n\n\n\n\n"]
        }
        response = requests.post(url, json=data, headers=headers)
        text = response.json()['output']['choices'][0]['text']
        print(text)

        return text

    else:
        # Anything else is served by the OpenAI-compatible endpoint, e.g. ornith.
        output = _compatible_call(model, query, temp)
        print(output)

        return output


def get_answers_with_image(model, api_key, query, img_path, temp, want_json=False) -> str:
    """
    Given a prompt and a rendered image: get the model's response.
    Used for the visual feedback that drives the refinement loop.
    """

    mime, encoded = _encode_image(img_path)

    if is_gemini(model):
        parts = [{"text": query}, {"inline_data": {"mime_type": mime, "data": encoded}}]
        return _gemini_call(model, api_key, parts, temp, want_json=want_json)

    if is_compatible_endpoint(model):
        content = [
            {"type": "text", "text": query},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
        ]
        return _compatible_call(model, content, temp)

    raise ValueError(
        f"Model '{model}' cannot take images. Use a gemini-* model or one served by "
        "the OpenAI-compatible endpoint for the vision model."
    )


def get_json_answer(model, api_key, query, img_path, temp=0.0) -> dict:
    """As get_answers_with_image, but parses a JSON object out of the reply."""

    raw = get_answers_with_image(model, api_key, query, img_path, temp, want_json=True)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Models commonly wrap JSON in a ```json fence or add a sentence around it.
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            return json.loads(raw[start:end + 1])
        raise
