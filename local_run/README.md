# Local run

A self-contained way to run Query2CAD on a normal Windows machine, using FreeCAD
locally and hosted models for the language and vision work. Nothing outside this
folder is modified: `src/` keeps the original implementation, and the prompts in
`src/prompts.py` are reused unchanged.

## Why this exists

The original pipeline (`src/run.py`) needs three things this setup replaces:

| Original | Problem on a laptop | Replacement here |
| --- | --- | --- |
| pyautogui driving the FreeCAD window | Needs a 1920x1080 screen, hardcoded pixel coordinates, uninterrupted window focus, and it reads errors off the clipboard | `freecad_runner.py` runs each macro headlessly in `freecadcmd.exe` and reads back a structured result |
| BLIP2 flan-t5-xxl captions | ~30 GB VRAM in 8-bit | Gemini vision caption |
| t2v_metrics VQAScore (clip-flant5-xl) | Several more GB of VRAM | Gemini vision score |
| `src/gui_run.py` | Imported by `src/run.py` but missing from the repository | not needed |

The algorithm is unchanged: plan the steps, generate a macro, execute it, repair
execution errors, then refine against visual feedback until it passes.

## Setup

```powershell
pip install -r local_run\requirements.txt
```

That is the whole dependency list -- both APIs are called over plain HTTP, and
FreeCAD runs through its own bundled Python.

Then edit `local_run\.env`:

```ini
FREECAD_CMD=D:\FreeCAD\bin\freecadcmd.exe
GEMINI_API_KEY=...            # https://aistudio.google.com/apikey
ORNITH_BASE_URL=...           # optional, OpenAI-compatible server
ORNITH_API_KEY=...
```

`.env` is gitignored and must stay that way -- it holds live credentials. Copy
`.env.example` when setting up somewhere new.

## Running

```powershell
# one query, typed in
python local_run\run_local.py --mode single

# one query, given directly
python local_run\run_local.py --mode single --query "A cube with side 10mm and a 3mm hole through the centre"

# the bundled dataset (57 queries in data/queries.txt)
python local_run\run_local.py --mode dataset --limit 5
```

Useful flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--code_model` | `gemini-2.5-flash` | Model that writes the macro |
| `--reasoning_model` | `gemini-2.5-flash` | Model that plans the steps |
| `--vision_model` | `gemini-2.5-flash` | Model that grades the render; must be `gemini-*` |
| `--vqa_threshold` | `0.8` | Score, 0-1, needed to accept a model |
| `--error_iterations` | `3` | Repair attempts per failed execution |
| `--refine_iterations` | `3` | Visual refinement rounds |
| `--timeout` | `300` | Seconds allowed per macro |
| `--human_feedback` | off | Confirm scores and captions yourself |
| `--start` / `--limit` | - | Slice of the dataset to run |

## Choosing models

Two backends are wired up, picked automatically from the model name:

* **`gemini-*`** goes to Google AI Studio and handles text *and* images.
* **anything else** (for example `ornith`) goes to the OpenAI-compatible server
  at `ORNITH_BASE_URL`.

The Ornith server does not accept image attachments, so it can generate code but
cannot grade renders. `--vision_model` therefore has to stay on Gemini:

```powershell
python local_run\run_local.py --mode single --code_model ornith --reasoning_model ornith
```

If Gemini's free tier starts returning 429s, the client backs off and retries on
its own; `GEMINI_THINKING_BUDGET=0` in `.env` makes calls faster and cheaper.

## Output

One folder per query, with steps numbered in the order they happened:

```
local_run/output/run_20260812_153011/
├── summary.json
└── query_00_cube-with-side-dimension-10mm/
    ├── 01_direct.FCMacro          generated macro
    ├── 01_direct.png              isometric render
    ├── 02_direct_repair1.FCMacro  only if execution failed
    ├── 03_refine1.FCMacro         only if the render did not pass
    └── run.json                   plan, scores, captions, errors per step
```

`run.json` records the outcome of every query: `passed_direct`,
`passed_refine_<n>`, `threshold_not_reached`, `no_executable_macro`,
`refinement_failed`, `stopped_by_user` or `crashed`.

## Files

| File | Role |
| --- | --- |
| `run_local.py` | Entry point and the generate/execute/repair/refine loop |
| `llm_router.py` | Picks a backend from the model name |
| `gemini_client.py` | Google AI Studio calls, text and image |
| `openai_compat.py` | OpenAI-compatible chat calls |
| `vision_feedback.py` | Scoring and captioning prompts |
| `freecad_runner.py` | Runs a macro in `freecadcmd.exe` and returns errors or a render |
| `_fc_driver.py` | Executes inside FreeCAD: runs the macro, captures the isometric view |

## Notes on the FreeCAD side

Rendering headlessly needs three things in order, all handled in `_fc_driver.py`:
`FreeCADGui.showMainWindow()`, a realised main window, and a pumped Qt event queue
before `saveImage`. Skip the event pump and every capture silently comes out as a
flat top view, whatever the camera was set to.

Macros are executed with `App`, `Gui`, `FreeCAD` and `FreeCADGui` already in scope,
matching how FreeCAD runs a macro from its own editor, and a document is opened in
advance so macros that assume one work unchanged. A macro that raises, produces no
document, or produces no visible geometry is reported as an error and sent back to
the model for repair.
