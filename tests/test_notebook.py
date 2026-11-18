import ast
import base64
import hashlib
import io
import json
import zipfile

import pytest

from moderation.common import ROOT


def test_notebook_cells_and_embedded_source_match_repository():
    path = ROOT / "notebooks" / "01_colab_training.ipynb"
    if not path.exists():
        pytest.skip(
            "Source-only Colab archive; run scripts/build_colab_notebook.py to regenerate the notebook."
        )
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert len({cell["id"] for cell in notebook["cells"]}) == len(notebook["cells"])
    archive_text = None
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            tree = ast.parse(cell["source"])
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "ARCHIVE_B64" for t in node.targets
                ):
                    archive_text = ast.literal_eval(node.value)
    assert archive_text
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(archive_text))) as archive:
        assert {
            "app.py", "config.json", "requirements.txt", "run.cmd", ".gitattributes", ".vscode/launch.json"
        }.issubset(archive.namelist())
        for name in archive.namelist():
            assert archive.read(name) == (ROOT / name).read_bytes(), f"Embedded source is stale: {name}"
    bootstrap = next(cell["source"] for cell in notebook["cells"] if "ARCHIVE_B64 =" in cell["source"])
    assert hashlib.sha256(base64.b64decode(archive_text)).hexdigest() in bootstrap
