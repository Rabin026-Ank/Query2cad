"""Query2CAD, wired for a local Windows run with FreeCAD + Gemini.

Same algorithm as src/run.py -- reason about steps, generate a macro, execute it,
repair errors, then refine against visual feedback until it passes -- with two
components swapped out so it runs on a normal laptop:

  * FreeCAD is driven headlessly (freecad_runner) instead of through pyautogui.
  * Gemini provides both the code generation and the visual feedback that the
    paper took from GPT-4/Llama, BLIP2 flan-t5-xxl and t2v_metrics VQAScore.

Every artefact of a run lands under local_run/output/<query folder>/.

Usage:
    python local_run/run_local.py --mode single
    python local_run/run_local.py --mode dataset --limit 5
"""

import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# The prompts are the substance of the paper, so they are reused as-is from src/.
# local_run stays first on the path: src/ has its own llm.py that pulls in the
# together and openai SDKs, which this setup deliberately does not install.
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from prompts import (  # noqa: E402
    get_code_prompt,
    get_error_prompt,
    get_feedback_reason_prompt,
    get_steps_prompt,
    get_vqa_prompt,
)

import llm_router as llm  # noqa: E402
from freecad_runner import find_freecad_cmd, run_macro  # noqa: E402
from vision_feedback import describe_render, score_render  # noqa: E402


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #

def load_env(path=os.path.join(HERE, ".env")):
    """Minimal .env reader, so the only dependency stays `requests`."""

    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# --------------------------------------------------------------------------- #
# Output layout
# --------------------------------------------------------------------------- #

class Workspace:
    """One folder per query, with steps numbered in the order they happened."""

    def __init__(self, root, index, query):
        slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")[:40].strip("-")
        self.dir = os.path.join(root, f"query_{index:02d}_{slug}" if slug else f"query_{index:02d}")
        os.makedirs(self.dir, exist_ok=True)
        self.step = 0
        self.log = {"index": index, "query": query.strip(), "steps": []}

    def next_paths(self, label):
        self.step += 1
        stem = os.path.join(self.dir, f"{self.step:02d}_{label}")
        return stem + ".FCMacro", stem + ".png"

    def record(self, **entry):
        self.log["steps"].append(entry)
        self.flush()

    def flush(self):
        with open(os.path.join(self.dir, "run.json"), "w", encoding="utf-8") as fh:
            json.dump(self.log, fh, indent=2)


CODE_FENCE = re.compile(r"```(?:python|py|FCMacro)?[ \t]*\n(.*?)(?:```|\Z)", re.S)


def extract_code(text):
    """Pull runnable Python out of a model reply.

    The prompts end on an open ```python fence, so replies come back either as a
    complete fenced block or as bare code that closes a fence it never opened.
    """

    match = CODE_FENCE.search(text)
    if match:
        return match.group(1).strip()
    return text.split("```")[0].strip()


def write_macro(code, path):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(code)


def get_queries(path):
    with open(path, "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def repair_loop(code, error, cfg, ws, label):
    """Feed execution errors back to the model until the macro runs, or give up."""

    for attempt in range(1, cfg.error_iterations + 1):
        print(f"  [repair {attempt}/{cfg.error_iterations}] {error.strip().splitlines()[-1][:120]}")
        reply = llm.ask(cfg.code_model, get_error_prompt(code, error), cfg.code_temperature)
        code = extract_code(reply)
        macro_path, img_path = ws.next_paths(f"{label}_repair{attempt}")
        write_macro(code, macro_path)
        error = run_macro(macro_path, img_path, timeout=cfg.timeout)
        ws.record(stage=f"{label}_repair{attempt}", macro=os.path.basename(macro_path),
                  image=None if error else os.path.basename(img_path), error=error)
        if error is None:
            return code, img_path, None
    return code, None, error


def process_query(index, query, cfg, out_root):
    ws = Workspace(out_root, index, query)
    print(f"\n=== Query {index}: {query.strip()}")
    print(f"    output -> {ws.dir}")

    # 1. Plan the modelling steps, then generate the macro.
    steps = llm.ask(cfg.reasoning_model, get_steps_prompt(query), cfg.reasoning_temperature)
    reply = llm.ask(cfg.code_model, get_code_prompt(query, steps), cfg.code_temperature)
    code = extract_code(reply)

    macro_path, img_path = ws.next_paths("direct")
    write_macro(code, macro_path)
    ws.log["steps_plan"] = steps

    # 2. Execute, repairing errors if needed.
    error = run_macro(macro_path, img_path, timeout=cfg.timeout)
    ws.record(stage="direct", macro=os.path.basename(macro_path),
              image=None if error else os.path.basename(img_path), error=error)

    if error is not None:
        code, img_path, error = repair_loop(code, error, cfg, ws, "direct")
        if error is not None:
            print("  Could not get an executable macro for this query. Skipping.")
            ws.log["outcome"] = "no_executable_macro"
            ws.flush()
            return ws.log

    # 3. Visual gate: does the render match the request?
    subject = llm.ask(cfg.reasoning_model, get_vqa_prompt(query), cfg.reasoning_temperature).strip()
    score, reason = score_render(img_path, query, cfg.vision_model)
    ws.record(stage="score", image=os.path.basename(img_path), subject=subject,
              score=score, reason=reason)

    if cfg.human_feedback and score < cfg.vqa_threshold:
        if input("  Is the model actually correct and the score wrong? (y/N): ").lower() == "y":
            score = 1.0

    if score >= cfg.vqa_threshold:
        print(f"  Passed with no refinement needed (score {score:.2f}).")
        ws.log["outcome"] = "passed_direct"
        ws.log["final_score"] = score
        ws.flush()
        return ws.log

    # 4. Refine against visual feedback.
    print("  Direct attempt is not good enough. Starting refinement...")
    caption = describe_render(img_path, cfg.vision_model)

    for round_no in range(1, cfg.refine_iterations + 1):
        if cfg.human_feedback:
            choice = input("  Use this caption (y), type your own (n), or stop refining (s)? ").lower()
            if choice == "s":
                ws.log["outcome"] = "stopped_by_user"
                break
            if choice == "n":
                caption = input("  Your feedback: ").strip()

        reply = llm.ask(cfg.code_model, get_feedback_reason_prompt(caption, query, code),
                        cfg.code_temperature)
        code = extract_code(reply)
        macro_path, img_path = ws.next_paths(f"refine{round_no}")
        write_macro(code, macro_path)

        error = run_macro(macro_path, img_path, timeout=cfg.timeout)
        ws.record(stage=f"refine{round_no}", macro=os.path.basename(macro_path),
                  image=None if error else os.path.basename(img_path), error=error,
                  caption_used=caption)

        if error is not None:
            code, img_path, error = repair_loop(code, error, cfg, ws, f"refine{round_no}")
            if error is not None:
                print("  Refinement produced no executable macro. Skipping query.")
                ws.log["outcome"] = "refinement_failed"
                break

        score, reason = score_render(img_path, query, cfg.vision_model)
        ws.record(stage=f"refine{round_no}_score", image=os.path.basename(img_path),
                  score=score, reason=reason)

        if score >= cfg.vqa_threshold:
            print(f"  Refinement succeeded on round {round_no}.")
            ws.log["outcome"] = f"passed_refine_{round_no}"
            ws.log["final_score"] = score
            break

        if round_no < cfg.refine_iterations:
            caption = describe_render(img_path, cfg.vision_model)
    else:
        print("  Ran out of refinement rounds below the threshold.")
        ws.log["outcome"] = "threshold_not_reached"

    ws.log.setdefault("final_score", score)
    ws.flush()
    return ws.log


def main():
    load_env()

    parser = argparse.ArgumentParser(description="Run Query2CAD locally with FreeCAD + Gemini")
    parser.add_argument("--mode", choices=["dataset", "single"], default="single",
                        help="Run the bundled dataset or a single typed query")
    parser.add_argument("--query", type=str, default=None,
                        help="The query for --mode single (prompted for if omitted)")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N dataset queries")
    parser.add_argument("--start", type=int, default=0, help="Dataset index to start from")
    parser.add_argument("--code_model", default=os.getenv("CODE_MODEL", "gemini-2.5-flash"),
                        help="gemini-* or a model on the Ornith endpoint, e.g. 'ornith'")
    parser.add_argument("--reasoning_model", default=os.getenv("REASONING_MODEL", "gemini-2.5-flash"))
    parser.add_argument("--vision_model", default=os.getenv("VISION_MODEL", "gemini-2.5-flash"),
                        help="Must be gemini-*; the Ornith server does not accept images")
    parser.add_argument("--code_temperature", type=float, default=0.2)
    parser.add_argument("--reasoning_temperature", type=float, default=0.8)
    parser.add_argument("--error_iterations", type=int, default=3)
    parser.add_argument("--refine_iterations", type=int, default=3)
    parser.add_argument("--vqa_threshold", type=float, default=0.8, help="Pass mark, 0-1")
    parser.add_argument("--timeout", type=int, default=300, help="Seconds allowed per macro")
    parser.add_argument("--human_feedback", action="store_true",
                        help="Ask for confirmation on scores and captions")
    parser.add_argument("--output", default=os.path.join(HERE, "output"))
    cfg = parser.parse_args()

    # Only demand the credentials the chosen models actually need.
    text_models = {cfg.code_model, cfg.reasoning_model}
    if any(llm.is_gemini(m) for m in text_models | {cfg.vision_model}) and not os.getenv("GEMINI_API_KEY"):
        sys.exit("No GEMINI_API_KEY found. Add it to local_run/.env")
    if any(not llm.is_gemini(m) for m in text_models) and not os.getenv("ORNITH_BASE_URL"):
        sys.exit("No ORNITH_BASE_URL found. Add it to local_run/.env")
    if not llm.is_gemini(cfg.vision_model):
        sys.exit(f"--vision_model must be a gemini-* model ('{cfg.vision_model}' cannot take images)")

    print(f"FreeCAD : {find_freecad_cmd()}")
    print(f"Models  : code={cfg.code_model} reasoning={cfg.reasoning_model} vision={cfg.vision_model}")

    if cfg.mode == "dataset":
        queries = get_queries(os.path.join(ROOT, "data", "queries.txt"))
        queries = queries[cfg.start:]
        if cfg.limit:
            queries = queries[:cfg.limit]
        offset = cfg.start
    else:
        queries = [cfg.query or input("Describe the CAD model you want to generate:\n> ")]
        offset = 0

    run_dir = os.path.join(cfg.output, time.strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    print(f"Output  : {run_dir}")

    summary = []
    for i, query in enumerate(queries):
        try:
            summary.append(process_query(offset + i, query, cfg, run_dir))
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as exc:
            print(f"  Query failed: {type(exc).__name__}: {exc}")
            summary.append({"index": offset + i, "query": query.strip(),
                            "outcome": "crashed", "error": str(exc)})

    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    passed = sum(1 for s in summary if str(s.get("outcome", "")).startswith("passed"))
    print(f"\n{passed}/{len(summary)} queries passed. Full run in {run_dir}")


if __name__ == "__main__":
    main()
