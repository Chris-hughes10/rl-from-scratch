"""Check that the shared modules in tutorial/lib/ match the notebooks that teach them.

Every top-level function, class and assignment in a module must be identical (as a syntax tree) to the
same-named definition in its notebook, so that importing the module gives exactly the code the reader studied.
Names that exist only in the module (module plumbing such as `start_gateway`) are listed, not compared.

Usage: .venv/bin/python tools/check_series.py
"""

import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SERIES = REPO / "tutorial/series"
NOTEBOOK_1 = "grpo_manual.ipynb"  # the new notebook 1; becomes 1_grpo_from_scratch.ipynb when it replaces it
NOTEBOOK_2 = "2_from_trainer_to_verl.ipynb"
MODULES = {  # module -> the notebook(s) that teach its code
    "arith_task.py": [NOTEBOOK_1],
    "rl_core.py": [NOTEBOOK_1, NOTEBOOK_2],  # notebook 2 §2.3-§2.5 teaches vllm_generate, save_for_inference, push_weights, plot_histories
    "verl_core.py": [NOTEBOOK_2],
    "recording_gateway.py": ["3_training_real_agents.ipynb"],
}


def definitions(src: str) -> dict[str, ast.AST]:
    found = {}
    for node in ast.parse(src).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found[node.name] = node
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            found[node.targets[0].id] = node
    return found


def notebook_definitions(name: str) -> dict[str, ast.AST]:
    found = {}
    for cell in json.load(open(SERIES / name))["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] != "code" or src.startswith("%%"):
            continue
        found.update(definitions(src))  # later cells win, as when the notebook runs
    return found


failures = 0
for module, notebooks in MODULES.items():
    in_module = definitions((REPO / "tutorial/lib" / module).read_text())
    in_notebook = {}
    for name in notebooks:
        in_notebook.update(notebook_definitions(name))  # later notebooks win
    notebook = " + ".join(notebooks)
    compared, module_only = [], []
    for name, node in in_module.items():
        if name not in in_notebook:
            module_only.append(name)
            continue
        if ast.dump(node) != ast.dump(in_notebook[name]):
            print(f"MISMATCH {module}:{name} differs from its definition in {notebook}")
            failures += 1
        compared.append(name)
    print(f"{module}: {len(compared)} definitions match {notebook}" + (f"; module only: {', '.join(module_only)}" if module_only else ""))

sys.exit(1 if failures else 0)
