"""Execute every notebook under docs/ and refresh its stored outputs.

The docs build (see docs/conf.py, nb_execution_mode = "off") renders whichever
outputs are already stored in each notebook rather than re-running them. This
test is the pre-flight check that those stored outputs are reproducible: it
re-executes each notebook and writes the result back in place, the same as
`jupyter nbconvert --execute --inplace`. A notebook only gets overwritten if
it runs cleanly, so a failing notebook is left untouched on disk and fails
the test.
"""

from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

DOCS_DIR = Path(__file__).parents[2] / "docs"

# Notebooks that download source data from S3 on a cold cache, or call a
# hosted physrisk API. Kept behind the same live_data gate used elsewhere in
# the suite so a plain `pytest` run doesn't require network access.
LIVE_DATA_NOTEBOOKS = {
    "user_guide/vulnerability/vulnerability_functions/wind_hazus/onboard.ipynb",
    "user_guide/vulnerability/vulnerability_functions/inundation_hazus/onboard.ipynb",
    "getting_started/asset_level_impacts.ipynb",
    "getting_started/hazard_indicators.ipynb",
    "getting_started/hazard_inventory.ipynb",
}


def _discover_notebooks():
    return sorted(
        path
        for path in DOCS_DIR.rglob("*.ipynb")
        if "_build" not in path.parts and ".ipynb_checkpoints" not in path.parts
    )


def _to_param(path: Path):
    rel = path.relative_to(DOCS_DIR).as_posix()
    marks = [pytest.mark.live_data("dev")] if rel in LIVE_DATA_NOTEBOOKS else []
    return pytest.param(path, id=rel, marks=marks)


NOTEBOOK_PARAMS = [_to_param(path) for path in _discover_notebooks()]


@pytest.mark.parametrize("notebook_path", NOTEBOOK_PARAMS)
def test_notebook_executes_cleanly(notebook_path: Path):
    nb = nbformat.read(notebook_path, as_version=4)
    client = nbclient.NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(notebook_path.parent)}},
        record_timing=False,
    )
    client.execute()
    # execution_count is set unconditionally by nbclient regardless of
    # record_timing; null it out (cell-level and inside execute_result
    # outputs) so re-running this test doesn't leave a diff on its own,
    # matching what the nbstripout pre-commit hook would otherwise do at
    # commit time.
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        cell["execution_count"] = None
        for output in cell.get("outputs", []):
            if "execution_count" in output:
                output["execution_count"] = None
    # Only reached on successful execution, so a failing notebook is left
    # untouched on disk rather than overwritten with a partial/errored result.
    nbformat.write(nb, notebook_path)
