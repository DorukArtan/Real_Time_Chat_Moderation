"""Exercise the notebook's real repair path without network access or training."""

import ast
import json
import os
import pathlib
import subprocess
import sys
import venv

import pytest

from moderation.common import ROOT


def setup_functions():
    notebook = ROOT / "notebooks" / "01_colab_training.ipynb"
    if not notebook.exists():
        pytest.skip("Source-only archive: rebuild the notebook before checking its setup cell.")
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    source = next(cell["source"] for cell in cells if "def create_experiment_environment(" in cell["source"])
    functions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)]
    namespace = {"os": os, "pathlib": pathlib, "subprocess": subprocess, "sys": sys}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "colab_setup_cell", "exec"), namespace)
    return namespace


def test_repairs_partial_venv_without_ensurepip(tmp_path):
    pytest.importorskip("pip")
    target = tmp_path / "partial experiment environment"
    # Reproduce the existing interpreter/pyvenv.cfg left by failed pip bootstrapping.
    venv.EnvBuilder(with_pip=False, system_site_packages=False).create(target)
    executable = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    assert subprocess.run([str(executable), "-m", "pip", "--version"], capture_output=True).returncode != 0

    functions = setup_functions()
    python = functions["create_experiment_environment"](target)
    assert "include-system-site-packages = true" in (target / "pyvenv.cfg").read_text()
    prefix = subprocess.check_output([python, "-c", "import sys; print(sys.prefix)"], text=True).strip()
    assert pathlib.Path(prefix).resolve() == target.resolve()
    # pip can manage this interpreter without installing into or changing the host.
    empty_requirements = tmp_path / "empty-requirements.txt"
    empty_requirements.write_text("")
    functions["setup_checked"](
        [
            sys.executable,
            "-m",
            "pip",
            "--python",
            python,
            "install",
            "--no-index",
            "-r",
            str(empty_requirements),
        ]
    )


def test_setup_failure_reports_original_error():
    functions = setup_functions()
    with pytest.raises(RuntimeError, match="original diagnostic"):
        functions["setup_checked"]([sys.executable, "-c", "import sys; sys.exit('original diagnostic')"])
