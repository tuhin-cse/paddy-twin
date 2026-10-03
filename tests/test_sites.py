"""Site layouts for Paddy Twin: the generated grid, site files, and the survey converter."""

import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import sites  # noqa: E402


def test_grid_layout():
    lay = sites.grid_layout(48)
    assert len(lay["fields"]) == 48 and len(lay["tubewells"]) == 2 and lay["loggers"] == []
    assert all(abs(f["area_m2"] - 900) < 1e-6 and len(f["poly"]) == 4 for f in lay["fields"])


def test_example_site_loads_with_loggers_inside_their_fields():
    lay = sites.load_site("example_irregular")
    assert lay["surveyed"] is False and len(lay["fields"]) > 50 and len(lay["loggers"]) > 5
    for lg in lay["loggers"]:
        assert sites.point_in_polygon(lg["x"], lg["z"], lay["fields"][lg["field"]]["poly"])
    assert {f["tw"] for f in lay["fields"]} == {0, 1, 2}


def test_unknown_site():
    with pytest.raises(KeyError):
        sites.load_site("../../etc/passwd")


def test_survey_converter(tmp_path):
    # two 20 m x 30 m plots near Jamalpur, a logger in the second, one tubewell
    lon0, lat0, dx, dz = 89.98, 25.2, 1 / 100_600, 1 / 111_000
    plot = lambda x0: [[lon0 + (x0 + a) * dx, lat0 + b * dz] for a, b in ((0, 0), (20, 0), (20, 30), (0, 30), (0, 0))]  # noqa: E731
    fields = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"id": f"P{i}"}, "geometry": {"type": "Polygon", "coordinates": [plot(25 * i)]}} for i in (1, 2)]}
    (tmp_path / "plots.geojson").write_text(json.dumps(fields))
    (tmp_path / "loggers.csv").write_text(f"id,lat,lon\nL1,{lat0 + 15 * dz},{lon0 + 60 * dx}\n")
    (tmp_path / "tw.csv").write_text(f"id,lat,lon,name\n7,{lat0 + 15 * dz},{lon0 - 10 * dx},Pump 7\n")
    name = "pytest_tmp_site"
    try:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "site_from_survey.py"), "--name", name, "--label", "test",
                        "--fields", str(tmp_path / "plots.geojson"), "--loggers", str(tmp_path / "loggers.csv"),
                        "--tubewells", str(tmp_path / "tw.csv")], check=True, capture_output=True)
        lay = sites.load_site(name)
        assert [f["id"] for f in lay["fields"]] == ["P1", "P2"]
        assert abs(lay["fields"][0]["area_m2"] - 600) < 15
        assert lay["loggers"][0]["field"] == 1 and lay["tubewells"][0]["name"] == "Pump 7"
    finally:
        (sites.SITES_DIR / f"{name}.geojson").unlink(missing_ok=True)


def test_site_imagery():
    lay = sites.load_site("example_irregular")
    img = lay["imagery"]
    assert img and len(img["dates"]) >= 3 and img["base"] == "/imagery/example_irregular/"
    assert img["center_xz"] == [0.0, 0.0] and not img["matches_plots"]      # synthetic plots on a real image
    for d in img["dates"]:
        assert (sites.IMAGERY_DIR / "example_irregular" / f"{d}.jpg").is_file()
    grid = sites.layout("grid", 48)
    assert grid["imagery"] and grid["imagery"]["base"] == "/imagery/_study_area/" and grid["imagery"]["center_xz"][0] > 0
