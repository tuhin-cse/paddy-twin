"""Logger placement planner (app/planner.py): adjacency, eligibility, the plan checks and the exports."""

import csv
import io
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "src"))
import planner  # noqa: E402
import sites  # noqa: E402


def test_grid_adjacency_and_strata():
    lay = sites.grid_layout(48)
    info = planner.site_info(lay)
    # neighbours in a row share a bund; plots across the channel do not
    assert 1 in info["adj"][0] and 12 not in info["adj"].get(0, {})
    assert set(info["strata"]) == {"tubewell", "channel position"}      # grid plots are all the same size
    assert sorted(set(info["strata"]["channel position"])) == ["head", "middle", "tail"]
    # every planned pipe is inside its plot
    assert all(sites.point_in_polygon(*info["pipe"][i], f["poly"]) for i, f in enumerate(lay["fields"]))


def test_suggest_respects_consent_locks_and_exclusions():
    lay = sites.load_site("example_irregular")
    info = planner.site_info(lay)
    assert "soil" in info["strata"] and "consent" not in info["strata"]
    no = [i for i, c in enumerate(info["consent"]) if c == "no"]
    assert no
    ok = [i for i in range(len(no) + 80) if i not in no]
    locked, excluded = [ok[3], ok[50]], ok[5:8]
    sel = planner.suggest(lay, 30, locked=locked, excluded=excluded)
    assert len(sel) == 30 and set(locked) <= set(sel) and not set(sel) & set(excluded + no)
    ev = planner.evaluate(lay, sel)
    assert ev["ineligible"] == [] and ev["missing"] == []
    assert ev["balance_vs_random"] >= 0.9                 # better balanced than random plans of the same size
    assert len(ev["pairs"]) >= 1 and ev["route"]["length_m"] > 0 and sorted(ev["route"]["order"]) == sel


def test_calibration_numbers():
    lay = sites.grid_layout(48)
    ev = planner.evaluate(lay, list(range(10)), p_fail=0.1)
    assert ev["calibration"]["p_enough"] == 0.0             # 10 loggers can never give 20
    assert ev["calibration"]["plan_size_for"]["0.9"] >= 20
    ev = planner.evaluate(lay, list(range(30)), p_fail=0.1)
    assert 0.95 < ev["calibration"]["p_enough"] <= 1 and ev["calibration"]["exp19"]["sp"] > 0.9
    assert planner._p_at_least(20, 0.0, 20) == pytest.approx(1.0)


def test_exports():
    lay = sites.load_site("example_irregular")
    sel = planner.suggest(lay, 22)
    rows = list(csv.DictReader(io.StringIO(planner.export_csv(lay, sel))))
    assert len(rows) == 22 and [int(r["visit_order"]) for r in rows] == list(range(1, 23))
    assert {r["field_id"] for r in rows} == {lay["fields"][i]["id"] for i in sel}
    gj = json.loads(planner.export_geojson(lay, sel, sites.SITES_DIR / "example_irregular.geojson"))
    lg = [f for f in gj["features"] if f["properties"]["kind"] == "logger"]
    assert len(lg) == 22 and all(f["properties"]["planned"] for f in lg)
    assert sum(f["properties"]["kind"] == "field" for f in gj["features"]) == len(lay["fields"])
