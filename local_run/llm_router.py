"""Model router: picks a backend from the model name.

  gemini-*  -> Google AI Studio  (text and images)
  anything else -> the OpenAI-compatible endpoint in .env, e.g. `ornith`

Image prompts are Gemini-only, since the Ornith server does not accept attachments.
"""

import os

import gemini_client
import openai_compat


def is_gemini(model):
    return model.startswith("gemini") or model in gemini_client.ALIASES


def ask(model, prompt, temperature=0.2):
    """Send a text prompt to whichever backend owns this model name."""

    if is_gemini(model):
        return gemini_client.ask(model, os.getenv("GEMINI_API_KEY", ""), prompt, temperature)
    return openai_compat.ask(model, prompt, temperature)


def ask_with_image(model, prompt, image_path, temperature=0.2, want_json=False):
    """Send a prompt plus an image. Requires a Gemini model."""

    if not is_gemini(model):
        raise ValueError(
            f"Model '{model}' cannot take images. Use a gemini-* model for --vision_model."
        )
    return gemini_client.ask_with_image(model, os.getenv("GEMINI_API_KEY", ""), prompt,
                                        image_path, temperature, want_json=want_json)


def ask_json_with_image(model, prompt, image_path, temperature=0.0):
    if not is_gemini(model):
        raise ValueError(
            f"Model '{model}' cannot take images. Use a gemini-* model for --vision_model."
        )
    return gemini_client.ask_json(model, os.getenv("GEMINI_API_KEY", ""), prompt,
                                  image_path=image_path, temperature=temperature)
