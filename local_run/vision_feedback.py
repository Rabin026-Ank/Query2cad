"""Visual feedback for the refinement loop, using Gemini instead of local models.

The paper's pipeline scores a render with t2v_metrics VQAScore (clip-flant5-xl)
and describes it with BLIP2 flan-t5-xxl. Together those need roughly 25-30 GB of
VRAM in 8-bit, which does not fit a 4 GB laptop GPU, and BLIP2's one-line captions
describe CAD renders poorly anyway.

Both roles are filled here by a Gemini vision model:
  score_render()    -> the VQA-style pass/fail gate
  describe_render() -> the caption fed back into the refinement prompt
"""

from llm_router import ask_json_with_image, ask_with_image

SCORE_PROMPT = """You are grading a CAD model that was generated from a text description.

The image is an isometric render of the model, shown in FreeCAD on a white background.

Requested design: {query}

Judge only whether the rendered geometry matches the requested design: the shape,
its features and the proportions implied by any stated dimensions. Ignore colour,
lighting, camera angle and overall scale.

Reply with JSON only:
{{"score": <float 0.0-1.0>, "reason": "<one short sentence>"}}

Use the full range: 1.0 is an exact match, 0.5 is the right family of shape with a
clear feature or proportion wrong, 0.0 is unrelated or empty geometry."""

DESCRIBE_PROMPT = """The image is an isometric render of a CAD model in FreeCAD.

Describe only what you actually see, in one short noun phrase, the way you would
complete the sentence "the CAD model generated resembles ...".

State the shape and any obvious features or proportions, for example "a flat
rectangular plate with a single circular hole near one edge" or "a cylinder that is
much wider than it is tall". Do not guess intent, do not mention CAD software,
colours or the background, and do not add any preamble.

Remember that an isometric view stretches the vertical direction: a cube looks
taller than it is wide in this projection. Only call out a proportion as unequal
when it is clearly beyond that effect."""


def score_render(image_path, query, model):
    """VQA-style gate: how well does the render match the request (0.0-1.0)?"""

    try:
        data = ask_json_with_image(model, SCORE_PROMPT.format(query=query.strip()), image_path)
        score = float(data.get("score", 0.0))
        reason = str(data.get("reason", "")).strip()
    except Exception as exc:
        # A grading failure must not abort the run; treat it as "needs refinement".
        print(f"  [vision] scoring failed ({exc}); treating as 0.0")
        return 0.0, "scoring failed"

    score = max(0.0, min(1.0, score))
    print(f"  [vision] score {score:.2f} -- {reason}")
    return score, reason


def describe_render(image_path, model):
    """Caption used as the feedback signal in the refinement prompt."""

    caption = ask_with_image(model, DESCRIBE_PROMPT, image_path, temperature=0.2)
    caption = caption.strip().strip('"').rstrip(".")
    print(f"  [vision] caption: {caption}")
    return caption
