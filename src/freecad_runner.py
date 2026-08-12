"""Headless execution of generated FreeCAD macros.

Backs utils.gui_sequence. Each macro runs in its own freecadcmd.exe process
together with _fc_driver.py, which executes it, renders an isometric view and
reports the outcome as JSON.
"""

import json
import os
import subprocess
import sys
import tempfile
from shutil import which

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER = os.path.join(HERE, "_fc_driver.py")

CANDIDATE_PATHS = [
    r"D:\FreeCAD\bin\freecadcmd.exe",
    r"C:\Program Files\FreeCAD\bin\freecadcmd.exe",
    r"C:\Program Files\FreeCAD 1.1\bin\freecadcmd.exe",
    r"C:\Program Files\FreeCAD 1.0\bin\freecadcmd.exe",
    "/usr/bin/freecadcmd",
    "/usr/local/bin/freecadcmd",
]


def find_freecad_cmd():
    """Locate the FreeCAD console executable: FREECAD_CMD, then usual spots, then PATH."""

    explicit = os.environ.get("FREECAD_CMD")
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(f"FREECAD_CMD points at a missing file: {explicit}")
        return explicit

    for path in CANDIDATE_PATHS:
        if os.path.isfile(path):
            return path

    found = which("freecadcmd") or which("FreeCADCmd")
    if found:
        return found

    raise FileNotFoundError(
        "Could not find freecadcmd. Set FREECAD_CMD in .env, for example "
        r"FREECAD_CMD=D:\FreeCAD\bin\freecadcmd.exe"
    )


def run_macro(macro_path, img_path, timeout=300, width=1050, height=675, verbose=True):
    """Execute one macro and capture its isometric render.

    Returns error text if the macro failed to produce geometry, or None on success.
    """

    macro_path = os.path.abspath(macro_path)
    img_path = os.path.abspath(img_path)
    os.makedirs(os.path.dirname(img_path), exist_ok=True)

    # A stale render from an earlier attempt must not be mistaken for this one.
    if os.path.exists(img_path):
        os.remove(img_path)

    handle, result_path = tempfile.mkstemp(suffix=".json", prefix="q2c_")
    os.close(handle)

    env = dict(os.environ)
    env.update({
        "Q2C_MACRO": macro_path,
        "Q2C_IMAGE": img_path,
        "Q2C_RESULT": result_path,
        "Q2C_WIDTH": str(width),
        "Q2C_HEIGHT": str(height),
    })

    try:
        proc = subprocess.run(
            [find_freecad_cmd(), DRIVER],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        os.unlink(result_path)
        return (f"Execution timed out after {timeout} seconds. The macro is probably stuck "
                "in an endless loop or an operation far too expensive to compute.")

    try:
        with open(result_path, "r", encoding="utf-8") as fh:
            outcome = json.load(fh)
    except Exception:
        # The driver never got far enough to report; fall back to process output.
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-25:]
        return (f"FreeCAD exited without reporting a result (exit code {proc.returncode}).\n"
                + "\n".join(tail))
    finally:
        if os.path.exists(result_path):
            os.unlink(result_path)

    error = outcome.get("error")
    if error:
        if verbose:
            print(f"  [freecad] failed at stage '{outcome.get('stage')}'")
        return error

    if not os.path.exists(img_path):
        return "FreeCAD reported success but no image was written."

    if verbose:
        print(f"  [freecad] rendered {outcome.get('objects')} object(s)")
    return None


if __name__ == "__main__":
    # Smoke test: python src/freecad_runner.py <macro> <image>
    err = run_macro(sys.argv[1], sys.argv[2])
    print("ERROR:\n" + err if err else "OK")
