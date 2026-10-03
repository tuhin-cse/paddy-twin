"""Write app/campaigns/demo_synthetic/: a SYNTHETIC campaign in the replay format (app/campaign.py).

A season simulated with simulator v2 (random-mismatch world, NISAR on) on the synthetic example site,
written out as the campaign files real data will arrive in: 2-hourly logger readings, the SmartSoil app log
(irrigations as pump hours, "water gone" reports), per-field satellite wet/dry calls, weather, transplanting
dates, measured pump discharge and weekly tubewell meter volumes. It also carries problems the campaign will
meet, listed in campaign.json: a logger that dies, one with a gap, one that drifts, and farmers who never use
the app. It is labelled synthetic everywhere; it exists to exercise replay mode before January 2027.
"""

import csv
import json
import pathlib
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from app import sites  # noqa: E402
from awdrt import kgml_paddy as kp  # noqa: E402
from awdrt import paddy_sim as ps  # noqa: E402

OUT = ROOT / "app" / "campaigns" / "demo_synthetic"
SITE = "example_irregular"
SEED = 2027


def write(name, header, rows):
    with open(OUT / name, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def main():
    inp = ps.load_inputs()
    lay = sites.load_site(SITE)
    n = len(lay["fields"])
    cfg = replace(ps.WORLDS["reality"], nisar=True, p_gone_reporter=0.1)
    sim = ps.simulate(cfg, n + n % 2, SEED, inp)
    T = sim["X"].shape[1]
    start = inp["day0"] + pd.Timedelta(days=int(inp["s0"]))
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    ids = [f["id"] for f in lay["fields"]]
    day = lambda k: (start + pd.Timedelta(days=int(k))).strftime("%Y-%m-%d")  # noqa: E731

    write("fields.csv", ["field_id", "transplant_date"], [[ids[f], day(np.floor(sim["tp"][f]))] for f in range(n)])
    discharge = {t["id"]: round(float(rng.uniform(90, 140)), 1) for t in lay["tubewells"]}
    write("tubewells.csv", ["tubewell_id", "discharge_m3h"], [[k, v] for k, v in discharge.items()])

    # loggers: 2-hourly readings (6-hourly simulated level, interpolated, 2 mm noise), with injected faults
    issues = []
    rows = []
    hours6 = np.arange(T * ps.PER_DAY) * 6.0
    hours2 = np.arange(0, T * 24, 2.0)
    for i, lg in enumerate(lay["loggers"]):
        h6 = sim["h6"][lg["field"]]
        ok = np.isfinite(h6)
        if ok.sum() < 2:
            continue
        lvl = np.interp(hours2, hours6[ok], h6[ok], left=np.nan, right=np.nan)
        lvl[hours2 < hours6[ok][0]] = np.nan
        lvl = lvl + 2.0 * rng.standard_normal(lvl.size)
        if i == 6:
            lvl[hours2 >= 55 * 24] = np.nan
            issues.append(f"logger {lg['id']} stops on {day(55)} (dead battery)")
        if i == 11:
            lvl[(hours2 >= 40 * 24) & (hours2 < 46 * 24)] = np.nan
            issues.append(f"logger {lg['id']} has no readings {day(40)} to {day(45)} (data gap)")
        if i == 19:
            lvl = lvl + np.clip(hours2 / 24 - 40, 0, None) * 1.5
            issues.append(f"logger {lg['id']} drifts upwards by 1.5 mm/day from {day(40)} (sensor drift)")
        lvl = np.clip(lvl, ps.PIPE_MIN, ps.PIPE_MAX)
        for h, v in zip(hours2, lvl):
            if np.isfinite(v):
                rows.append([lg["id"], (start + pd.Timedelta(hours=float(h))).strftime("%Y-%m-%dT%H:%M"), round(float(v), 1)])
    write("loggers.csv", ["logger_id", "time", "level_mm"], rows)

    # SmartSoil app log: logged irrigations as pump hours; "water gone" reports; 25 % of farmers never log
    X = sim["X"]
    no_app = rng.random(n) < 0.25
    issues.append(f"{int(no_app.sum())} of {n} farmers never use the app (no log events)")
    rows = []
    for f in range(n):
        if no_app[f]:
            continue
        disc = discharge[lay["tubewells"][lay["fields"][f]["tw"]]["id"]]
        for k in range(T):
            if X[f, k, kp.CH["irr_logged"]] > 0 and sim["irr_mm"][f, k] > 0:
                hours = sim["irr_mm"][f, k] * lay["fields"][f]["area_m2"] / 1000 / disc
                rows.append([ids[f], day(k), "irrigated", round(float(hours), 2), ""])
            if X[f, k, kp.CH["gone"]] > 0:
                rows.append([ids[f], day(k), "water_gone", "", ""])
    write("app_log.csv", ["field_id", "date", "event", "pump_hours", "depth_mm"], rows)

    # satellite calls per field (the Layer 1 detector's output), Sentinel-2 with LAI
    rows = []
    for k in range(T):
        for s in ("s2", "s1", "nisar"):
            wet, dry = X[:n, k, kp.CH[f"{s}_wet"]] > 0, X[:n, k, kp.CH[f"{s}_dry"]] > 0
            for f in np.flatnonzero(wet | dry):
                rows.append([day(k), s, ids[f], int(wet[f]), round(float(X[f, k, kp.CH["lai_obs"]] * 5), 2) if s == "s2" else ""])
    write("satellite.csv", ["date", "sensor", "field_id", "wet", "lai"], rows)

    s0 = int(inp["s0"])
    write("weather.csv", ["date", "et0_mm", "rain_mm", "tmean_c"],
          [[day(k), round(float(inp["et0"][s0 + k]), 2), round(float(inp["rain"][s0 + k]), 1), round(float(inp["tmean"][s0 + k]), 1)]
           for k in range(min(T + 31, len(inp["et0"]) - s0))])

    # weekly metered volumes: true irrigation x field area, plus 10 % conveyance loss
    rows = []
    for t_i, t in enumerate(lay["tubewells"]):
        fs = [f for f in range(n) if lay["fields"][f]["tw"] == t_i]
        for wk in range(0, T, 7):
            m3 = sum(sim["irr_mm"][f, wk:wk + 7].sum() * lay["fields"][f]["area_m2"] / 1000 for f in fs) * 1.1
            rows.append([t["id"], day(min(wk + 6, T - 1)), round(float(m3), 1)])
    write("meters.csv", ["tubewell_id", "date", "m3"], rows)

    meta = {"label": "Demo campaign (synthetic, not field data)", "site": SITE, "start": day(0), "end": day(T - 1),
            "synthetic": True, "source": "scripts/make_demo_campaign.py: simulator v2, random-mismatch world, NISAR on",
            "demo_injected_issues": issues}
    (OUT / "campaign.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {OUT}: {n} fields, {len(lay['loggers'])} loggers, {T} days; issues: {issues}")


if __name__ == "__main__":
    main()
