# Driver executed *inside* FreeCAD by freecad_runner.run_macro.
#
# Boots the FreeCAD GUI layer under freecadcmd.exe, executes the generated macro,
# renders an isometric view and writes an outcome JSON the parent process reads
# back. Paths arrive through environment variables because freecadcmd does not
# forward extra command line arguments to the script.

import json
import os
import sys
import traceback

MACRO_PATH = os.environ["Q2C_MACRO"]
IMAGE_PATH = os.environ["Q2C_IMAGE"]
RESULT_PATH = os.environ["Q2C_RESULT"]
IMG_W = int(os.environ.get("Q2C_WIDTH", "1050"))
IMG_H = int(os.environ.get("Q2C_HEIGHT", "675"))

result = {"error": None, "image": None, "objects": 0, "stage": "startup"}


def finish(code=0):
    with open(RESULT_PATH, "w", encoding="utf-8") as fh:
        json.dump(result, fh)
    sys.exit(code)


# --- Boot the GUI layer -------------------------------------------------------
# saveImage needs a realised main window and a pumped Qt event queue. Without
# both, camera changes never reach the renderer and every capture comes out as a
# flat top view.
try:
    import FreeCAD
    import FreeCADGui

    FreeCADGui.showMainWindow()
    FreeCADGui.getMainWindow().show()

    from PySide import QtGui

    _app = QtGui.QApplication.instance()

    def pump(times=25):
        for _ in range(times):
            _app.processEvents()

    pump()
except Exception:
    result["error"] = "FreeCAD GUI initialisation failed:\n" + traceback.format_exc()
    finish(1)


# --- Execute the generated macro ---------------------------------------------
result["stage"] = "execute"
try:
    with open(MACRO_PATH, "r", encoding="utf-8") as fh:
        source = fh.read()
except Exception:
    result["error"] = "Could not read the macro file:\n" + traceback.format_exc()
    finish(1)

# Pre-create a document so macros that assume one is already open work unchanged.
FreeCAD.newDocument("Query2CAD")
pump(5)

namespace = {
    "__name__": "__main__",
    "__file__": MACRO_PATH,
    "App": FreeCAD,
    "FreeCAD": FreeCAD,
    "Gui": FreeCADGui,
    "FreeCADGui": FreeCADGui,
}

try:
    exec(compile(source, MACRO_PATH, "exec"), namespace)
except SystemExit:
    pass
except BaseException:
    result["error"] = traceback.format_exc()
    finish(0)

pump()

# --- Check that something was actually built ----------------------------------
result["stage"] = "inspect"
doc = FreeCAD.ActiveDocument
if doc is None:
    result["error"] = (
        "The macro ran without raising an exception but no FreeCAD document exists. "
        "Create a document and add the shape to it, for example with Part.show(shape)."
    )
    finish(0)

try:
    doc.recompute()
except Exception:
    pass
pump()

visible = 0
for obj in doc.Objects:
    try:
        shape = getattr(obj, "Shape", None)
        if shape is not None and not shape.isNull():
            visible += 1
        elif getattr(obj, "Mesh", None) is not None:
            visible += 1
    except Exception:
        continue
result["objects"] = visible

if visible == 0:
    result["error"] = (
        "The macro ran without raising an exception but produced no visible geometry. "
        "Make sure the resulting shape is added to the document (for example with "
        "Part.show(shape)) and that the document is recomputed."
    )
    finish(0)

# --- Capture the isometric render --------------------------------------------
result["stage"] = "render"
try:
    gui_doc = FreeCADGui.activeDocument()
    if gui_doc is None:
        FreeCADGui.setActiveDocument(doc.Name)
        gui_doc = FreeCADGui.activeDocument()

    view = gui_doc.activeView()
    view.viewIsometric()
    pump()
    view.fitAll()
    pump()
    view.saveImage(IMAGE_PATH, IMG_W, IMG_H, "White")
    result["image"] = IMAGE_PATH
except Exception:
    result["error"] = "Rendering the model failed:\n" + traceback.format_exc()
    finish(0)

result["stage"] = "done"
finish(0)
