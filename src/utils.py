import os, sys, requests, re
import time
import logging
from prompts import *
from llm import *
from freecad_runner import run_macro

# Prompts for the visual feedback loop. These replace the two local models the
# paper used for the same two jobs: t2v_metrics VQAScore for the pass/fail gate
# and BLIP2 flan-t5-xxl for the caption fed back into refinement.

VQA_SCORE_PROMPT = """You are grading a CAD model that was generated from a text description.

The image is an isometric render of the model, shown in FreeCAD on a white background.

Requested design: {query}

Judge only whether the rendered geometry matches the requested design: the shape,
its features and the proportions implied by any stated dimensions. Ignore colour,
lighting, camera angle and overall scale.

Reply with JSON only:
{{"score": <float 0.0-1.0>, "reason": "<one short sentence>"}}

Use the full range: 1.0 is an exact match, 0.5 is the right family of shape with a
clear feature or proportion wrong, 0.0 is unrelated or empty geometry."""

CAPTION_PROMPT = """The image is an isometric render of a CAD model in FreeCAD.

Describe only what you actually see, in one short noun phrase, the way you would
complete the sentence "the CAD model generated resembles ...".

State the shape and any obvious features or proportions, for example "a flat
rectangular plate with a single circular hole near one edge" or "a cylinder that is
much wider than it is tall". Do not guess intent, do not mention CAD software,
colours or the background, and do not add any preamble.

Remember that an isometric view stretches the vertical direction: a cube looks
taller than it is wide in this projection. Only call out a proportion as unequal
when it is clearly beyond that effect."""


def get_queries(path):
    """
    Gets all the user queries. Assumes that they are stored in a .txt file and each new line is a new query.
    """

    with open(path, 'r', encoding='utf-8') as file:
        queries = file.readlines()

    return queries

def write_macro(code, macro_path):
    """
    Writes the generated macro in the specified path.
    """

    os.makedirs(os.path.dirname(macro_path), exist_ok=True)
    with open(macro_path, 'w', encoding='utf-8') as file:
        file.write(code)

def remove_backticks(text):
    """
    Post processing on generated code output -- Extracts the code from the model reply.

    The prompts end on an open ```python fence, so a reply arrives either as a
    complete fenced block (chat models) or as bare code that closes a fence it
    never opened (completion models).
    """

    fenced = re.search(r"```(?:python|py|FCMacro)?[ \t]*\n(.*?)(?:```|\Z)", text, re.S)
    if fenced:
        return fenced.group(1).strip()

    pattern = r"```"
    matches = re.finditer(pattern, text)
    position = [m.start() for m in matches] #get the location of ```

    try:
        end_position = position[0]
        code = text[0:end_position].strip()

    except:
        code = text.strip()

    return code

def gui_sequence(macro_code_path, img_path):
    """
    Runs the entire sequence -- opening FreeCAD, running generated code, capturing the isometric image,
    returning the error code if there is any.

    The macro is executed headlessly through freecadcmd rather than by driving the
    FreeCAD window, so the run does not depend on screen resolution, window focus
    or an idle desktop.
    """

    logging.info('Executing macro %s', macro_code_path)
    error_msg = run_macro(macro_code_path, img_path,
                          timeout=int(os.getenv("FREECAD_TIMEOUT", "300")))

    return error_msg

def get_executable_code(gen_code, error, error_iter, model, api_key, temp, query_idx, base_url, direct_code=False, refined_code=False, refined_idx = 0):
    """
    Tries to get an executable code
    """

    updated_code = gen_code
    if direct_code:
        for i in range(1, error_iter + 1):
            error_prompt = get_error_prompt(updated_code, error)
            updated_code = get_answers(model, api_key, error_prompt, temp, base_url)
            updated_code = remove_backticks(updated_code)
            macro_path = f"results/code/query_{query_idx}_direct_attempt_{i}.FCMacro"
            write_macro(updated_code, macro_path)
            img_path = f"results/images/query_{query_idx}_direct_attempt_{i}.png"
            error_msg = gui_sequence(macro_path, img_path)

            if error_msg is not None:
                error = error_msg
                continue
            else:
                break

    else:
        for i in range(1, error_iter + 1):
            error_prompt = get_error_prompt(updated_code, error)
            updated_code = get_answers(model, api_key, error_prompt, temp, base_url)
            updated_code = remove_backticks(updated_code)
            macro_path = f"results/code/query_{query_idx}_refined_{refined_idx}_attempt_{i}.FCMacro"
            write_macro(updated_code, macro_path)
            img_path = f"results/images/query_{query_idx}_refined_{refined_idx}_attempt_{i}.png"
            error_msg = gui_sequence(macro_path, img_path)

            if error_msg is not None:
                error = error_msg
                continue
            else:
                break

    return error_msg, i, updated_code

def get_vqa_score(img_path, user_query, model):
    """
    Gets the VQA score between the generated model and user query.

    `model` is the name of the vision model doing the grading.
    """

    api_key = os.getenv("GEMINI_API_KEY", "")
    prompt = VQA_SCORE_PROMPT.format(query=user_query.strip())

    try:
        data = get_json_answer(model, api_key, prompt, img_path)
        score = float(data.get("score", 0.0))
        reason = str(data.get("reason", "")).strip()
    except Exception as exc:
        # A grading failure must not abort the run; treat it as "needs refinement".
        print(f"  [vqa] scoring failed ({exc}); treating as 0.0")
        return 0.0

    score = max(0.0, min(1.0, score))
    print(f"  [vqa] score {score:.2f} -- {reason}")

    return score

def get_captions(img_path, processor, model, human_feedback):
    """
    Gets the caption of the image

    `processor` is retained for signature compatibility with the BLIP2 version
    and is unused. `model` is the name of the vision model.
    """

    api_key = os.getenv("GEMINI_API_KEY", "")
    caption = get_answers_with_image(model, api_key, CAPTION_PROMPT, img_path, 0.2)
    caption = caption.strip().strip('"').rstrip('.')

    print('System generated caption: ', caption)

    if human_feedback != False:
        proceed = input("Do you want to proceed with the system generated caption? Type fp for vqa score false negative (y/n/fn): ")

        if proceed == 'y' or proceed == 'Y':
            caption = caption

        elif proceed == 'n' or proceed == 'N':
            caption = input("Please enter the feedback: ")

        elif proceed == 'fn' or proceed == 'FN':
            caption = proceed

    else:
        caption = caption

    return caption

def get_refined_outputs(captions, user_query, prev_code, refine_iter, model, api_key, temp,
                        query_idx, error_iter, vqa_model, vqa_thresh, processor, caption_model, base_url, human_feedback):
    """
    Performs iterative refinement and gets the best possible output for a given user query
    """

    refined_code = prev_code
    for i in range(refine_iter):
        print('Final Caption: ', captions)
        if captions == 'fn' or captions == 'FN':
            print('Stopping the refinements as asked by user... Moving to next query')
            break

        feedback_reason_prompt = get_feedback_reason_prompt(captions, user_query, refined_code)
        refined_code = get_answers(model, api_key, feedback_reason_prompt, temp, base_url)
        refined_code = remove_backticks(refined_code)
        macro_path = f"results/code/query_{query_idx}_refined_{i}_attempt_0.FCMacro"
        write_macro(refined_code, macro_path)
        img_path = f"results/images/query_{query_idx}_refined_{i}_attempt_0.png"
        error_msg = gui_sequence(macro_path, img_path)
        print('error_msg', error_msg)
        if error_msg is not None:
            error, success_idx, refined_code = get_executable_code(refined_code, error_msg, error_iter, model, api_key, temp, query_idx, base_url, refined_code=True, refined_idx=i)
            if error is None:
                img_path_for_captions = f"results/images/query_{query_idx}_refined_{i}_attempt_{success_idx}.png"
            else:
                print("Refinement failed... Skipping to next query")
                break

        else:
            img_path_for_captions = img_path

        # Check VQA score
        vqa_score = get_vqa_score(img_path_for_captions, user_query, vqa_model)

        if vqa_score < vqa_thresh and i != refine_iter - 1:
            print("Doing another round of refinement...")
            captions = get_captions(img_path_for_captions, processor, caption_model, human_feedback)

        elif vqa_score < vqa_thresh and i == refine_iter - 1:
            print("Could not refine enough to cross the VQA threshold that was set within the defined refinement iterations...")

        elif vqa_score >= vqa_thresh:
            print(f"Refinement was successful for query {query_idx} in refine attempt {i}...")
            break
