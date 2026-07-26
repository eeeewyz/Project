from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = PROJECT_ROOT / "scripts" / "create_notebook.py"


def test_generator_writes_clean_five_section_notebook(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(GENERATOR)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    notebook_path = tmp_path / "notebooks" / "rag_pipeline_exploration.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    markdown = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )
    expected_sections = [
        "# 1. Project objective and architecture",
        "# 2. Load the synthetic data",
        "# 3. FAQ semantic retrieval",
        "# 4. Product filters and FAISS retrieval",
        "# 5. Measured evaluation",
    ]

    assert all(section in markdown for section in expected_sections)
    assert notebook.metadata["kernelspec"] == {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    assert all(
        cell.outputs == [] and cell.execution_count is None
        for cell in notebook.cells
        if cell.cell_type == "code"
    )
    assert "python -m evaluation.run_evaluation" in markdown
    assert not any(
        term in str(notebook).casefold()
        for term in (
            "course" + "ra",
            "gra" + "der",
            "unit" + "tests",
            "dlai" + ".link",
            "jov" + "yan",
        )
    )
