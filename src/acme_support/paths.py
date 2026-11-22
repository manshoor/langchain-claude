"""Where the project's files live.

The package is installed in editable mode from src/, so the repo root is two
folders above this file.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "docs"
DATA_DIR = ROOT / "evals" / "data"
SCRIPTS_DIR = ROOT / "scripts"
DB_PATH = ROOT / "chat.db"  # one file for 04, 05 and 06, as before the move


def load_script(name: str) -> ModuleType:
    """Import a numbered script, such as "05_support_bot", for the evals.

    A name that starts with a digit can't be imported with `import`, and
    scripts/ is not a package. A script loaded once is reused.
    """
    if name in sys.modules:
        return sys.modules[name]

    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
