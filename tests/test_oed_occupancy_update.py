"""
Utility test for regenerating OED_OCCUPANCY_CODES from the upstream source.

Run manually when the upstream CSV may have changed:

    RUN_MANUAL_TESTS=1 pytest tests/test_oed_occupancy_update.py -s
"""

import csv
import io
import os

import pytest
import requests


OED_OCCUPANCY_CSV_URL = (
    "https://raw.githubusercontent.com/OasisLMF/ODS_OpenExposureData"
    "/main/OpenExposureData/OccupancyValues.csv"
)

_run_manual = os.environ.get("RUN_MANUAL_TESTS", "").lower() in ("1", "true", "yes")


def _fetch_occupancy_codes(url: str) -> dict:
    text = requests.get(url).text
    reader = csv.DictReader(io.StringIO(text))
    result = {}
    for row in reader:
        try:
            code = int(row["OED Code"].strip())
        except ValueError:
            continue
        result[code] = row["Name"].strip()
    return result


def _format_dict_as_source(codes: dict) -> str:
    lines = ["OED_OCCUPANCY_CODES: Dict[int, str] = {"]
    for code, label in sorted(codes.items()):
        lines.append(f'    {code}: "{label}",')
    lines.append("}")
    return "\n".join(lines)


@pytest.mark.skipif(
    not _run_manual, reason="manual update utility — set RUN_MANUAL_TESTS=1 to run"
)
def test_update_oed_occupancy_codes():
    """
    Fetches the current OccupancyValues.csv from OasisLMF/ODS_OpenExposureData and
    prints the updated OED_OCCUPANCY_CODES dict body for pasting into
    src/physrisk/data/static/oed_occupancy.py.
    """
    from physrisk.data.static.oed_occupancy import OED_OCCUPANCY_CODES

    upstream = _fetch_occupancy_codes(OED_OCCUPANCY_CSV_URL)

    added = {k: v for k, v in upstream.items() if k not in OED_OCCUPANCY_CODES}
    removed = {k: v for k, v in OED_OCCUPANCY_CODES.items() if k not in upstream}
    changed = {
        k: (OED_OCCUPANCY_CODES[k], upstream[k])
        for k in upstream
        if k in OED_OCCUPANCY_CODES and upstream[k] != OED_OCCUPANCY_CODES[k]
    }

    if not (added or removed or changed):
        print("\nSnapshot is already up to date — no changes needed.")
        return

    print(f"\nAdded ({len(added)}):   {list(added.items())}")
    print(f"Removed ({len(removed)}): {list(removed.items())}")
    print(f"Changed ({len(changed)}): {list(changed.items())}")
    print("\n--- paste into src/physrisk/data/static/oed_occupancy.py ---\n")
    print(_format_dict_as_source(upstream))
