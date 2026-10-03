"""Replay mode: campaign files (app/campaign.py) and the model run on them (engine.run_replay)."""

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "src"))
from app import campaign, sites  # noqa: E402


def test_demo_campaign_loads():
    lay = sites.load_site("example_irregular")
    c = campaign.load("demo_synthetic", lay)
    assert c["n"] == len(lay["fields"]) and c["T"] == 91 and c["meta"]["synthetic"] is True
    assert c["logged"].sum() > 500 and (c["sat"]["s2"] >= 0).sum() > 1000
    assert set(c["logger6"]) == {lg["id"] for lg in lay["loggers"]}
    lai = campaign.daily_lai(c, __import__("awdrt.paddy_sim", fromlist=["x"]).lai_curve)
    assert lai.shape == (c["n"], c["T"]) and np.all(lai >= 0)


def test_unknown_campaign():
    with pytest.raises(KeyError):
        campaign.load("../../etc", sites.load_site("example_irregular"))


def test_observed_next_dry_censoring():
    import engine
    T = 6
    # 6-hourly readings: drains 40 -> 0 over the end of day 1; an irrigation (+60 mm) on day 4
    L6 = np.array([60, 55, 50, 45, 40, 30, 20, 10, -5, -20, -30, -40, -50, -60, -70, -80, 10, 70, 65, 60, 55, 50, 45, 40, 35], float)
    logged = np.zeros(T, bool)
    D = engine._observed_next_dry(L6, logged, np.zeros(T + 30), T)
    assert D[0] == pytest.approx(1.0)         # end of day 0 at 40 mm; below the surface 4 readings later
    assert D[2] == 0.0                         # already below the surface
    assert np.isnan(D[3])                      # the irrigation (a rise of > 10 mm) comes before drying


def test_replay_run():
    import engine
    from awdrt import paddy_sim as ps
    r = engine.run_replay("demo_synthetic", {}, ps.load_inputs())
    assert r["mode"] == "replay" and r["campaign"]["synthetic"] and r["n"] == 134
    st = r["replay_stats"]
    assert st["logger_field_days"] > 1000 and 0.55 < st["ponded_bacc"] <= 1
    # fields without a logger have no truth; logger fields do
    lf = {lg["field"] for lg in r["loggers"]}
    assert all(all(v is None for v in r["truth"][f]) for f in range(r["n"]) if f not in lf)
    assert any(h["stopped"] for h in r["campaign"]["logger_health"])          # the demo's dead logger
    with pytest.raises(ValueError):
        engine.ask_farmer(r["run"], 0, 50)
