import os, sys, requests, argparse
from utils import *
from prompts import *
from llm import *
from freecad_runner import find_freecad_cmd


def load_env(path=None):
    """
    Reads key=value pairs from the .env file at the repository root.

    Written out rather than taken from python-dotenv so the system needs only
    `requests` to run.
    """

    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")

    if not os.path.isfile(path):
        return

    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_3d(code_model, reasoning_model, error_iter, refine_iter, code_temp, reasoning_temp,
           code_api_key, reasoning_api_key, mode, vqa_model, vqa_threshold, human_feedback, base_url,
           query=None):

    print(f"FreeCAD: {find_freecad_cmd()}")
    print(f"Models : code={code_model} reasoning={reasoning_model} vqa={vqa_model}")

    if mode == "dataset":
        dataset_queries = get_queries('data/queries.txt')
    else:
        # A single query is still a list: iterating a bare string would walk it
        # one character at a time.
        dataset_queries = [query or input("Describe the CAD model that you want to generate:\n")]

    macro_store_path = 'results/code'

    # `processor` belonged to the BLIP2 captioner; the vision model needs no
    # local state, so it is passed through as None.
    processor = None

    # Iterate through all queries
    for idx, query in enumerate(dataset_queries):
        print(f"Starting query number {idx}...")
        # Get direct response
        direct_steps_prompt = get_steps_prompt(query)
        direct_steps = get_answers(reasoning_model, reasoning_api_key, direct_steps_prompt, reasoning_temp, base_url)
        direct_code_prompt = get_code_prompt(query, direct_steps)
        direct_code = get_answers(code_model, code_api_key, direct_code_prompt, code_temp, base_url)
        direct_code = remove_backticks(direct_code)
        direct_code_macro_file_path = f"{macro_store_path}/query_{idx}_direct_attempt_0.FCMacro"
        write_macro(direct_code, direct_code_macro_file_path)

        # Execute the macro and capture the isometric render
        img_path = f"results/images/query_{idx}_direct_attempt_0.png"
        error_msg = gui_sequence(direct_code_macro_file_path, img_path)

        # Get an executable code
        if error_msg is not None:
            error, success_idx, new_code = get_executable_code(direct_code, error_msg, error_iter, code_model, code_api_key, code_temp, idx, base_url, direct_code=True)
            if error is None:
                img_path_for_captions = f"results/images/query_{idx}_direct_attempt_{success_idx}.png"
                code_for_refinement = new_code
            else:
                print("Could not get an executable code for this query... Skipping to next query")
                continue

        else:
            img_path_for_captions = img_path
            code_for_refinement = direct_code

        # VQA score check
        prompt_for_vqa = get_vqa_prompt(query)
        query_for_vqa = get_answers(reasoning_model, reasoning_api_key, prompt_for_vqa, reasoning_temp, base_url)
        vqa_score = get_vqa_score(img_path_for_captions, query_for_vqa, vqa_model)
        print(vqa_score)

        if human_feedback != False:
            fp = input("Is the output correct but VQA score made a mistake? (y/n): ")

            if fp == 'y' or fp == 'Y':
                vqa_score = 1

            elif fp == 'n' or fp == 'N':
                vqa_score = vqa_score

        # Refine if VQA score check is not passed
        if vqa_score < vqa_threshold:
            print("Directly generated code is not correct... Beginning refinement...")
            # Get captions

            caption = get_captions(img_path_for_captions, processor, vqa_model, human_feedback)

            # Get the best possible code
            get_refined_outputs(caption, query, code_for_refinement, refine_iter, code_model, code_api_key, code_temp, idx,
                                error_iter, vqa_model, vqa_threshold, processor, vqa_model, base_url, human_feedback)

        else:
            print("Stopping criteria is reached with direct model output. No need of refinement...")



if __name__ == "__main__":
    load_env()

    args = argparse.ArgumentParser()
    args.add_argument("--code_gen_model", type=str, default=os.getenv("CODE_MODEL", "ornith"))
    args.add_argument("--reasoning_model", type=str, default=os.getenv("REASONING_MODEL", "ornith"))
    args.add_argument("--vqa_model", type=str, default=os.getenv("VISION_MODEL", "ornith"),
                      help="Model that grades and captions the render")
    args.add_argument("--error_iterations", type=int, default=3)
    args.add_argument("--refine_iterations", type=int, default=3)
    args.add_argument("--code_gen_temperature", type=float, default=0.2)
    args.add_argument("--reasoning_temperature", type=float, default=0.8)
    args.add_argument("--code_gen_api_key", type=str, default="")
    args.add_argument("--reasoning_api_key", type=str, default="")
    args.add_argument("--mode", type=str, help="Inferencing on dataset or single inference (dataset or single)", default="dataset")
    args.add_argument("--query", type=str, default=None, help="The query for --mode single (prompted for if omitted)")
    args.add_argument("--vqa_threshold", type=float, help="Between 0 and 1", default=0.9)
    args.add_argument("--human_feedback", type=str, default=False)

    args = args.parse_args()

    base_url = None

    # Resolve the credentials each chosen model needs.
    for role in ("code_gen", "reasoning"):
        model = getattr(args, "code_gen_model" if role == "code_gen" else "reasoning_model")
        key_attr = f"{role}_api_key"

        if getattr(args, key_attr):
            continue

        if is_gemini(model):
            setattr(args, key_attr, os.getenv("GEMINI_API_KEY", ""))

        elif model in ("chatgpt", "gpt4-turbo"):
            setattr(args, key_attr, os.getenv("PROXY_API_KEY", ""))
            base_url = os.getenv("PROXY_BASE_URL")

        elif model in ("codellama", "llama3"):
            setattr(args, key_attr, os.getenv("TOGETHER_API_KEY", ""))

        else:
            # Served by the OpenAI-compatible endpoint, which reads its own
            # credentials from ORNITH_BASE_URL / ORNITH_API_KEY.
            setattr(args, key_attr, "")

    get_3d(args.code_gen_model, args.reasoning_model, args.error_iterations, args.refine_iterations, args.code_gen_temperature, args.reasoning_temperature,
           args.code_gen_api_key, args.reasoning_api_key, args.mode, args.vqa_model, args.vqa_threshold, args.human_feedback, base_url,
           args.query)
