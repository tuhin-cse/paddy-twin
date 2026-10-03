"""Data quality control (app/qc.py): each check on small synthetic inputs, and the demo campaign's injected faults."""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "src"))
import qc  # noqa: E402

T, PD = 60, qc.PER_DAY


def tiny(readings):
    """A one-field, one-logger campaign with the given 6-hourly logger readings (length T * 4 + 1)."""
    lay = {"fields": [{"id": "F1", "area_m2": 1000.0, "tw": 0}], "tubewells": [{"id": "1"}],
           "loggers": [{"id": "L1", "field": 0}]}
    c = {"n": 1, "T": T, "start": pd.Timestamp("2024-01-15"), "rain": np.zeros(T + 32), "logger6": {"L1": np.asarray(readings, float)},
         "logged": np.zeros((1, T), bool), "gone": np.zeros((1, T), bool), "irr_mm": np.zeros((1, T)),
         "sat": {s: np.full((1, T), -1, np.int8) for s in ("nisar", "s2", "s1")}, "meters": []}
    return c, lay, np.ones((1, T), bool)


def wave(seed=0):
    """A clean AWD-like sawtooth (irrigate to 60 mm, fall 15 mm/day) with 2 mm noise."""
    k = np.arange(T * PD + 1)
    return 60 - 15 * ((k / PD) % 12) + 2 * np.random.default_rng(seed).standard_normal(k.size)


def kinds(readings, q50=None):
    c, lay, season = tiny(readings)
    q = q50 if q50 is not None else np.asarray(readings, float)[PD::PD][None, :T]
    # the tiny campaign has no satellite calls and no app log: keep the logger checks only
    return [i["kind"] for i in qc.check(c, lay, season, q, [])["issues"] if i["logger"]]


def test_clean_logger_has_no_issues():
    assert kinds(wave()) == []


def test_stopped_gap_flat_spike_range():
    r = wave(); r[40 * PD:] = np.nan
    assert "stopped" in kinds(r)
    r = wave(); r[20 * PD:23 * PD] = np.nan
    assert kinds(r) == ["gap"]
    r = wave(); r[30 * PD:33 * PD] = -40.0
    assert "flatline" in kinds(r, wave()[PD::PD][None, :T])
    r = wave(); r[100] += 120
    assert "spike" in kinds(r, wave()[PD::PD][None, :T])
    r = wave(); r[50] = 400
    assert "out_of_range" in kinds(r, wave()[PD::PD][None, :T])


def test_drift_against_the_model():
    clean = wave()
    drift = clean + np.clip(np.arange(clean.size) / PD - 25, 0, None) * 2.0
    ks = kinds(drift, clean[PD::PD][None, :T])
    assert ks == ["drift"]
    d = np.arange(T, dtype=float)
    fit = qc.hinge_drift(d, np.clip(d - 25, 0, None) * 2.0 + np.random.default_rng(1).normal(0, 5, T))
    assert abs(fit["onset"] - 25) <= 3 and 1.6 < fit["slope"] < 2.4


def test_runs():
    assert qc._runs(np.array([0, 1, 1, 0, 1], bool)) == [(1, 2), (4, 4)]


def test_demo_campaign_faults_found():
    import engine
    from awdrt import paddy_sim as ps
    r = engine.run_replay("demo_synthetic", {}, ps.load_inputs())
    iss = r["qc"]["issues"]
    found = {(i["kind"], i["logger"]) for i in iss}
    assert {("stopped", "L07"), ("gap", "L12"), ("drift", "L20")} <= found
    # no other logger is flagged as faulty
    assert {i["logger"] for i in iss if i["severity"] in ("high", "medium") and i["logger"]} == {"L07", "L12", "L20"}
    gap = next(i for i in iss if i["kind"] == "gap")
    assert (gap["start_date"], gap["end_date"]) == ("2024-02-24", "2024-02-29")
    assert next(i for i in iss if i["kind"] == "stopped")["start_date"] == "2024-03-10"
    assert any(i["kind"] == "no_app" and i["title"].startswith("34 of 134") for i in iss)
    md = engine.qc_report(r["run"], "md", 5)
    assert md.startswith("# Data quality report") and "L07" not in md.split("## Season so far")[0]  # week 6 is before L07 dies
    assert engine.qc_report(r["run"], "csv").startswith("severity,kind,logger")
    with pytest.raises(KeyError):
        engine.qc_report("nope")
