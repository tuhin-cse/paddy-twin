"""Experiment mode: one comparison over several simulated villages, with means, ranges and paired differences.

One app run is one village: one draw of fields, farmers, weather noise and satellite errors. A number from one run
is an anecdote. An experiment runs setting A and setting B on the same villages (seeds seed0, seed0 + 1, ...), so
every difference is paired. For each measure it reports the mean of A and of B with a 95 % confidence interval
over villages (Student t), the mean paired difference B - A with its interval, and how many villages B did better in.

Measures per village (as in the app's season summary), optionally with the methane comparison (Experiment 26)
and the advice counterfactual (Experiment 25; slow). Results export as CSV (one row per village and setting)
and as a Markdown summary that states the settings and seeds, so app figures can be cited and reproduced.
"""

import csv
import io
import math
import time

import numpy as np
from scipy import stats

try:
    from . import engine
except ImportError:  # imported as a top-level module (tests)
    import engine

# key: (label, unit, better: +1 higher, -1 lower, 0 neither, group)
MEASURES = {
    "ponded_bacc": ("Ponded or not, balanced accuracy", "", 1, "model"),
    "next_dry_err": ("Next-dry forecast error (mean absolute)", "d", -1, "model"),
    "alert_precision": ("Irrigate-by-tomorrow alerts right", "", 1, "model"),
    "alert_recall": ("Fields drying by tomorrow that got an alert", "", 1, "model"),
    "sp_err": ("Leak-rate error (median)", "%", -1, "model"),
    "n3_precision": ("Doubted Sentinel-2 passes that were wrong", "", 1, "model"),
    "nisar_gain_bacc": ("NISAR update: gain in ponded accuracy", "", 1, "model"),
    "nisar_gain_dry": ("NISAR update: change in next-dry error", "d", -1, "model"),
    "s2_wrong": ("Sentinel-2 passes wrong", "", -1, "sensors"),
    "s1_wrong": ("Sentinel-1 passes wrong", "", -1, "sensors"),
    "nisar_wrong": ("NISAR passes wrong", "", -1, "sensors"),
    "water_mm": ("Irrigation water per field", "mm", 0, "village"),
    "ponded_days": ("Days ponded per field", "d", 0, "village"),
    "mh_balanced": ("Methane: regimes the model gets right (balanced)", "", 1, "methane"),
    "mh_cf_recall": ("Methane: continuously flooded fields the model finds", "", 1, "methane"),
    "mh_model_err": ("Methane: model's village total, absolute error", "%", -1, "methane"),
    "mh_regional_gap": ("Methane: fields the regional default misjudges (H5b)", "", 0, "methane"),
    "cf_water": ("Advice vs practice: water", "%", -1, "advice"),
    "cf_stress": ("Advice vs practice: stress days per field", "d", -1, "advice"),
    "cf_methane": ("Advice vs practice: methane", "%", -1, "advice"),
}
VARY = ("model", "nisar_update", "world", "site", "leak", "et0_scale", "rain_scale", "cloud", "share_logging", "p_awd",
        "p_topup", "pump_bias", "nisar_acc", "nisar_assumed", "n_fields")
MAX_VILLAGES = 30


def _num(a):
    return np.array([np.nan if v is None else v for v in a], float)


def season_measures(R):
    """The season summary's measures from a run_season result."""
    T, n = R["T"], R["n"]
    truth = np.array([_num(r) for r in R["truth"]])
    pw = np.array([_num(r) for r in R["belief"]["p_wet"]])
    dp = np.array([_num(r) for r in R["dry_pred"]])
    dt = np.array([_num(r) for r in R["dry_true"]])
    ok = np.isfinite(truth) & np.isfinite(pw)
    m = {"ponded_bacc": engine._bacc(truth[ok] > 0, pw[ok] > 0.5)}
    al = ok & np.isfinite(dp) & (dp <= R["alert_days"])
    m["alert_precision"] = float(np.mean(np.nan_to_num(dt[al], nan=99) <= R["alert_days"])) if al.any() else None
    need = ok & (truth > 0) & np.isfinite(dt) & (dt <= R["alert_days"])
    m["alert_recall"] = float(np.mean(np.isfinite(dp[need]) & (dp[need] <= R["alert_days"]))) if need.any() else None
    both = ok & (truth > 0) & np.isfinite(dp) & np.isfinite(dt)
    # the mean, not the app's median: forecasts and truth sit on a 0.1-day grid, so a median moves in 0.1-day steps
    # and hides small paired differences
    m["next_dry_err"] = float(np.mean(np.abs(dp[both] - dt[both]))) if both.any() else None
    sp = [(F["sp_model"], F["sp_true"]) for F in R["fields"] if F["sp_true"]]
    m["sp_err"] = float((math.exp(np.median([abs(math.log(a / b)) for a, b in sp])) - 1) * 100) if sp else None
    for s in ("s2", "s1", "nisar"):
        P = [p for p in R["passes"] if p["s"] == s and p["truth"] is not None]
        m[f"{s}_wrong"] = float(np.mean([p["wet"] != p["truth"] for p in P])) if P else None
    fl = [p for p in R["passes"] if p["s"] == "s2" and p["flag"] and p["truth"] is not None]
    m["n3_precision"] = float(np.mean([p["wet"] != p["truth"] for p in fl])) if fl else None
    u = R["nisar_update"]
    m["nisar_gain_bacc"] = (u["ponded_bacc_after"] - u["ponded_bacc_before"]) if u.get("ponded_bacc_after") is not None else None
    m["nisar_gain_dry"] = (u["dry_err_after"] - u["dry_err_before"]) if u.get("dry_err_after") is not None else None
    m["water_mm"] = float(np.mean([F["irr_total"] for F in R["fields"]]))
    m["ponded_days"] = float(np.nansum(truth > 0) / n)
    return m


def run(spec, progress=None):
    """spec: {base: app params, vary: key, a: value, b: value, villages: k, seed0: int, methane: bool,
    counterfactual: bool, cf: {...}}. Returns rows (one per village and setting) and the summary."""
    vary = spec.get("vary", "nisar_update")
    if vary not in VARY:
        raise ValueError(f"cannot vary {vary!r}")
    k = int(min(max(int(spec.get("villages", 5)), 2), MAX_VILLAGES))
    seed0 = int(spec.get("seed0", 1))
    base = {kk: v for kk, v in (spec.get("base") or {}).items() if kk in engine.DEFAULTS and kk not in ("seed", vary)}
    arms = {"A": spec.get("a"), "B": spec.get("b")}
    settings = {a: engine.clean_params({**base, vary: v}) for a, v in arms.items()}
    if settings["A"][vary] == settings["B"][vary]:
        raise ValueError("A and B are the same setting")
    inp = engine.ps.load_inputs()
    rows, t0 = [], time.time()
    total = 2 * k
    for i in range(k):
        seed = seed0 + i
        for a in ("A", "B"):
            R = engine.run_season({**settings[a], "seed": seed}, inp)
            m = season_measures(R)
            if spec.get("methane"):
                M = engine.run_methane(R["run"], {})["sources"]
                m.update({"mh_balanced": M["model"]["vs_truth"]["regime_balanced"], "mh_cf_recall": M["model"]["vs_truth"]["cf_recall"],
                          "mh_model_err": abs(M["model"]["vs_truth"]["total_err"]) * 100,
                          "mh_regional_gap": M["regional"]["vs_truth"]["half_gap_share"]})
            if spec.get("counterfactual") and R["mode"] == "simulation":
                C = engine.run_counterfactual(R["run"], spec.get("cf") or {})["arms"]
                P, Mo = C["practice"], C["model"]
                m.update({"cf_water": (Mo["total_m3"] / P["total_m3"] - 1) * 100, "cf_stress": Mo["stress_days"] - P["stress_days"],
                          "cf_methane": (Mo["methane"]["kg_ha"] / P["methane"]["kg_ha"] - 1) * 100})
            rows.append({"village": i + 1, "seed": seed, "arm": a, "value": settings[a][vary], **m})
            if progress:
                progress(len(rows), total, time.time() - t0)
    return {"vary": vary, "a": settings["A"][vary], "b": settings["B"][vary], "villages": k, "seed0": seed0,
            "base": {kk: v for kk, v in settings["A"].items() if kk not in ("seed", vary)},
            "methane": bool(spec.get("methane")), "counterfactual": bool(spec.get("counterfactual")),
            "rows": rows, "summary": summarise(rows), "seconds": round(time.time() - t0, 1)}


def _ci(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if len(x) == 0:
        return None, None, None, 0
    m = float(x.mean())
    if len(x) < 2:
        return m, None, None, 1
    h = float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x)))
    return m, m - h, m + h, len(x)


def summarise(rows):
    out = []
    for key, (label, unit, better, group) in MEASURES.items():
        A = {r["village"]: r.get(key) for r in rows if r["arm"] == "A"}
        B = {r["village"]: r.get(key) for r in rows if r["arm"] == "B"}
        if all(v is None for v in list(A.values()) + list(B.values())):
            continue
        pairs = [(A[v], B[v]) for v in A if v in B and A[v] is not None and B[v] is not None]
        d = [b - a for a, b in pairs]
        ma, la, ha, na = _ci(A.values())
        mb, lb, hb, nb = _ci(B.values())
        md, ld, hd, nd = _ci(d)
        same = bool(d) and all(abs(x) < 1e-12 for x in d)   # the setting does not touch this measure
        wins = None if better == 0 or not d or same else int(sum((x > 1e-12) if better > 0 else (x < -1e-12) for x in d))
        out.append({"key": key, "label": label, "unit": unit, "better": better, "group": group,
                    "a": ma, "a_lo": la, "a_hi": ha, "b": mb, "b_lo": lb, "b_hi": hb, "diff": md, "diff_lo": ld, "diff_hi": hd,
                    "n": nd, "b_wins": wins, "same": same,
                    "clear": ld is not None and (ld > 0 or hd < 0)})   # the paired 95 % interval excludes zero
    return out


def to_csv(res):
    keys = [k for k in MEASURES if any(k in r for r in res["rows"])]
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["village", "seed", "arm", res["vary"], *keys])
    for r in res["rows"]:
        w.writerow([r["village"], r["seed"], r["arm"], r["value"], *[("" if r.get(k) is None else round(r[k], 4)) for k in keys]])
    return out.getvalue()


def _fmt(v, unit):
    if v is None:
        return "–"
    if unit == "":
        return f"{v:.3f}"
    return f"{v:.2f} {unit}" if unit == "d" else f"{v:.1f} {unit}"


def to_markdown(res):
    base = ", ".join(f"{k} = {v}" for k, v in res["base"].items())
    lines = [f"# Paddy Twin experiment: {res['vary']} = {res['a']} (A) against {res['b']} (B)", "",
             f"{res['villages']} simulated villages (seeds {res['seed0']}–{res['seed0'] + res['villages'] - 1}), each run with A and B "
             f"(paired). Other settings: {base}. Simulator v2, KGML-Paddy; every number is simulation.", "",
             "| Measure | A (95 % CI) | B (95 % CI) | B − A (95 % CI) | B better in |", "|---|---|---|---|---|"]
    for s in res["summary"]:
        wins = "same" if s["same"] else "–" if s["b_wins"] is None else f"{s['b_wins']} of {s['n']}"
        ci = lambda m, lo, hi: f"{_fmt(m, s['unit'])} ({_fmt(lo, s['unit'])} to {_fmt(hi, s['unit'])})" if lo is not None else _fmt(m, s["unit"])  # noqa: E731
        lines.append(f"| {s['label']} | {ci(s['a'], s['a_lo'], s['a_hi'])} | {ci(s['b'], s['b_lo'], s['b_hi'])} | "
                     f"{ci(s['diff'], s['diff_lo'], s['diff_hi'])}{' **' if s['clear'] else ''} | "
                     f"{wins} |")
    lines += ["", "** the paired 95 % interval excludes zero.",
              f"Reproduce: POST /api/experiment with vary = {res['vary']}, a = {res['a']}, b = {res['b']}, villages = {res['villages']}, "
              f"seed0 = {res['seed0']}, and the settings above."]
    return "\n".join(lines) + "\n"
