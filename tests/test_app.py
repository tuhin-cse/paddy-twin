"""Paddy Twin web app: the season and farmer-question endpoints (needs fastapi and httpx)."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from app.server import app  # noqa: E402


def test_season_and_ask():
    c = TestClient(app)
    r = c.post("/api/season", json={"n_fields": 24, "leak": 8, "cloud": 0.5, "seed": 3})
    assert r.status_code == 200
    s = r.json()
    n, T = s["n"], s["T"]
    assert n == 24 and len(s["truth"]) == n and len(s["truth"][0]) == T
    assert len(s["belief"]["p_wet"]) == n and all(len(row) == T for row in s["irr"])
    assert len(s["lai"]) == n and max(max(row) for row in s["lai"]) > 0.5  # true LAI drives the 3D rice
    assert s["params"]["leak"] == 8 and s["params"]["cloud"] == 0.5
    # three satellites; NISAR is checked against the water balance but is not a model input
    assert {p_["s"] for p_ in s["passes"]} == {"s2", "s1", "nisar"}
    assert s["sensors"]["nisar"]["model_input"] is False and s["sensors"]["s1"]["model_input"] is True
    assert all(p_["d"] in s["sensors"][p_["s"]]["days"] for p_ in s["passes"])
    # NISAR passes update the evening-before advice as answers (Experiment 24), unless switched off
    u = s["nisar_update"]
    assert u["enabled"] and u["n"] > 0 and 0.5 <= u["ponded_bacc_after"] <= 1
    assert any(p_.get("update") for p_ in s["passes"] if p_["s"] == "nisar")
    off = c.post("/api/season", json={"n_fields": 24, "leak": 8, "cloud": 0.5, "seed": 3, "nisar_update": False}).json()
    assert off["nisar_update"]["n"] == 0 and off["belief"]["p_wet"] != s["belief"]["p_wet"]
    # ask a farmer on a day the field is planted
    f, d = 0, next(d for d in range(T) if s["truth"][0][d] is not None)
    a = c.post("/api/ask", json={"run": s["run"], "field": f, "day": d + 5, "kind": "depth"}).json()
    assert set(a) >= {"answer", "before", "after", "truth_h"} and 0 <= a["after"]["p_wet"] <= 1
    assert c.post("/api/ask", json={"run": "nope", "field": 0, "day": 40}).status_code == 404


def test_season_on_a_site_file():
    c = TestClient(app)
    assert any(st["name"] == "example_irregular" for st in c.get("/api/sites").json())
    s = c.post("/api/season", json={"site": "example_irregular", "seed": 2}).json()
    assert s["n"] == len(s["layout"]["fields"]) == len(s["fields"]) and s["layout"]["surveyed"] is False
    lg = s["loggers"][0]
    assert len(lg["reading"]) == s["T"] and 0 <= lg["field"] < s["n"]
    assert all(f["area_m2"] > 100 for f in s["fields"])


def test_replay_endpoint():
    c = TestClient(app)
    camps = c.get("/api/campaigns").json()
    assert any(x["name"] == "demo_synthetic" and x["synthetic"] for x in camps)
    r = c.post("/api/replay", json={"campaign": "demo_synthetic", "nisar_update": False}).json()
    assert r["mode"] == "replay" and r["nisar_update"]["n"] == 0 and len(r["loggers"]) > 0
    assert c.post("/api/replay", json={"campaign": "nope"}).status_code == 404


def test_planner_endpoints():
    c = TestClient(app)
    s = c.post("/api/plan/site", json={"site": "example_irregular"}).json()
    assert len(s["eligible"]) == 134 and "soil" in s["strata"] and len(s["current"]) > 0
    ev = c.post("/api/plan/suggest", json={"site": "example_irregular", "n": 25, "locked": [3]}).json()
    assert ev["n"] == 25 and 3 in ev["fields"]
    ev2 = c.post("/api/plan/evaluate", json={"site": "example_irregular", "fields": ev["fields"]}).json()
    assert ev2["balance_mean"] == ev["balance_mean"]
    r = c.post("/api/plan/export", json={"site": "example_irregular", "fields": ev["fields"], "format": "csv"})
    assert r.status_code == 200 and r.text.startswith("logger_id,field_id")
    assert c.post("/api/plan/export", json={"site": "grid", "n_fields": 24, "fields": [1], "format": "geojson"}).status_code == 400
    assert c.post("/api/plan/site", json={"site": "nope"}).status_code == 404


def test_counterfactual_endpoint():
    c = TestClient(app)
    s = c.post("/api/season", json={"n_fields": 24, "seed": 4}).json()
    r = c.post("/api/counterfactual", json={"run": s["run"], "lead": 3, "ask": True})
    assert r.status_code == 200
    d = r.json()
    assert set(d["arms"]) == {"practice", "model", "regional", "et_only", "perfect"} and d["settings"]["lead"] == 3
    P, M = d["arms"]["practice"], d["arms"]["model"]
    assert len(M["field"]["water_mm"]) == 24 and len(M["cum_m3"]) == s["T"] and M["asked"] > 0 and P["asked"] == 0
    assert abs(P["total_m3"] - sum(sum(row) * f["area_m2"] / 1000 for row, f in zip(P["irr"], s["layout"]["fields"]))) < 0.01 * P["total_m3"] + 50
    assert c.post("/api/counterfactual", json={"run": "nope"}).status_code == 404


def test_methane_endpoint():
    c = TestClient(app)
    s = c.post("/api/season", json={"n_fields": 24, "seed": 6}).json()
    m = c.post("/api/methane", json={"run": s["run"], "min_dry": 2, "regional": "aggregated"}).json()
    assert set(m["sources"]) == {"truth", "model", "satellite", "regional"}
    t = m["sources"]["truth"]
    assert len(t["regime"]) == 24 and sum(t["counts"].values()) == 24 and t["total_lo"] < t["total_kg"] < t["total_hi"]
    assert m["sources"]["regional"]["counts"]["aggregated"] == 24 and 0 <= m["sources"]["model"]["vs_truth"]["regime_balanced"] <= 1
    assert c.post("/api/methane", json={"run": "nope"}).status_code == 404


def test_experiment_job():
    import time
    c = TestClient(app)
    r = c.post("/api/experiment", json={"base": {"n_fields": 24}, "vary": "nisar_update", "a": False, "b": True, "villages": 2})
    assert r.status_code == 200
    jid = r.json()["id"]
    for _ in range(240):
        j = c.get(f"/api/experiment/{jid}").json()
        if j["status"] != "running":
            break
        time.sleep(0.5)
    assert j["status"] == "done", j
    res = j["result"]
    assert len(res["rows"]) == 4 and {r_["arm"] for r_ in res["rows"]} == {"A", "B"}
    S = {s_["key"]: s_ for s_ in res["summary"]}
    assert S["water_mm"]["same"] and S["s2_wrong"]["same"]          # the NISAR update does not change the village
    assert not S["ponded_bacc"]["same"] and S["ponded_bacc"]["n"] == 2
    assert c.get(f"/api/experiment/{jid}/export.csv").text.startswith("village,seed,arm,nisar_update")
    assert "paired" in c.get(f"/api/experiment/{jid}/export.md").text
    bad = c.post("/api/experiment", json={"vary": "seed", "villages": 2}).json()["id"]
    for _ in range(20):
        j = c.get(f"/api/experiment/{bad}").json()
        if j["status"] != "running":
            break
        time.sleep(0.2)
    assert j["status"] == "failed"


def test_ask_with_a_given_answer():
    c = TestClient(app)
    s = c.post("/api/season", json={"n_fields": 24, "seed": 8}).json()
    f, d = 0, next(d for d in range(s["T"]) if s["truth"][0][d] is not None) + 10
    none = c.post("/api/ask", json={"run": s["run"], "field": f, "day": d, "kind": "none"}).json()
    assert none["answer"] is None and none["before"] == none["after"] and 0 <= none["before"]["p_dry_tomorrow"] <= 1
    lot = c.post("/api/ask", json={"run": s["run"], "field": f, "day": d, "kind": "depth", "answer": 3}).json()
    dry = c.post("/api/ask", json={"run": s["run"], "field": f, "day": d, "kind": "depth", "answer": 0}).json()
    assert lot["answer_code"] == 3 and dry["answer_code"] == 0
    assert lot["after"]["p_wet"] >= dry["after"]["p_wet"]                   # "a lot" of water is more water than "none"
    assert lot["after"]["dry_80"][1] >= dry["after"]["dry_80"][1]
    assert c.post("/api/ask", json={"run": s["run"], "field": f, "day": d, "kind": "depth", "answer": 7}).status_code == 400


def test_progress_reporting():
    c = TestClient(app)
    assert c.get("/api/progress/never-started").json() == {"waiting": True}
    s = c.post("/api/season", json={"n_fields": 24, "seed": 9, "_token": "t-season"})
    assert s.status_code == 200 and "_token" not in s.json()["params"]
    p = c.get("/api/progress/t-season").json()
    assert p["done"] and p["frac"] == 1.0 and len(p["stages"]) == 5 and p["stage"] == 4
    r = c.post("/api/replay", json={"campaign": "demo_synthetic", "_token": "t-replay"})
    assert r.status_code == 200
    p = c.get("/api/progress/t-replay").json()
    assert p["done"] and p["stages"][0] == "Reading the campaign files"
