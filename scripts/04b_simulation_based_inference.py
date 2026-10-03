"""Experiment 4b: simulation-based inference (SBI) of field S&P.

The closed-form estimator in hydro.py assumes every wet run starts with an
irrigation and every observation is right. Misclassified dates break that, and
its 90 % intervals under-cover (Experiment 4). Here the physics stays the same
but inference is amortised: simulate many fields with S&P from the prior, run
them through the same observation system (real S2 calendar and clouds,
Experiment-1 error rates, S1 after canopy closure, app logs), summarise each
observation sequence, and train quantile regressors from summaries to S&P.
Observation errors, censoring and farmer behaviour are then absorbed by
construction rather than assumed away.

Train 8,000 fields, test on the same 1,000 fields as Experiment 4.
"""

import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
tw = __import__("04_field_leakage_twin")
hydro = tw.hydro
OUT = tw.OUT


def runs(days, state):
    """Lengths (days) of observed wet runs and dry runs, first-to-last obs."""
    wet, dry = [], []
    if len(days) == 0:
        return wet, dry
    s0, t0 = state[0], days[0]
    for k in range(1, len(days) + 1):
        if k == len(days) or state[k] != s0:
            (wet if s0 == 1 else dry).append(days[k - 1] - t0)
            if k < len(days):
                s0, t0 = state[k], days[k]
    return wet, dry


def features(seasons, use_date, use_hours, use_gone, rng):
    f = {k: [] for k in ("n_obs", "frac_wet", "trans_per_day", "wet_run", "dry_run", "gap",
                         "et0", "rain", "irr_to_dry", "irr_to_lastwet", "h0", "irr_interval", "irr_to_gone")}
    for s in seasons:
        d, st = s["days"], s["state"]
        if len(d) < 2:
            continue
        span = d[-1] - d[0]
        w, dr = runs(d, st)
        f["n_obs"].append(len(d))
        f["frac_wet"].append(st.mean())
        f["trans_per_day"].append((np.diff(st) == -1).sum() / max(span, 1))
        f["wet_run"].append(np.mean(w) if w else 0)
        f["dry_run"].append(np.mean(dr) if dr else 0)
        f["gap"].append(np.mean(np.diff(d)))
        f["et0"].append(tw.et0_mean(d[0], d[-1]))
        f["rain"].append(tw.rain_sum(d[0], d[-1]) / max(span, 1))
        if use_date and s["irr_t"].size:
            to_dry, to_last = [], []
            for tau in s["irr_t"]:
                after = d > tau
                if after.any():
                    first_dry = d[after & (st == 0)]
                    to_dry.append(first_dry[0] - tau if first_dry.size else np.nan)
                    wets = d[after & (st == 1)]
                    wets = wets[wets < (first_dry[0] if first_dry.size else np.inf)]
                    to_last.append(wets[-1] - tau if wets.size else 0.0)
            f["irr_to_dry"].append(np.nanmedian(to_dry) if to_dry else np.nan)
            f["irr_to_lastwet"].append(np.nanmedian(to_last) if to_last else np.nan)
            f["irr_interval"].append(np.median(np.diff(s["irr_t"])) if s["irr_t"].size > 1 else np.nan)
        if use_hours and s["irr_h"].size:
            f["h0"].append(np.mean(s["irr_h"]))
        if use_gone and s["gone_t"].size:
            rep = rng.random(s["gone_t"].size) < 0.5
            dt_ = []
            for g in s["gone_t"][rep]:
                prev = s["irr_t"][s["irr_t"] < g]
                if prev.size:
                    dt_.append(g + rng.uniform(0, 1) - prev[-1])
            f["irr_to_gone"].append(np.median(dt_) if dt_ else np.nan)
    return {k: (np.nanmean(v) if len(v) and np.isfinite(v).any() else np.nan) for k, v in f.items()}


def build(n_fields, seed0, obs_days, obs_valid, rates):
    rows = []
    for n in range(n_fields):
        fr = np.random.default_rng(seed0 + n)
        sp = float(np.clip(fr.lognormal(np.log(5.0), 0.6), 0.5, 30))
        cache = {}
        for name, (ns, sar, d_, h_, g_) in tw.SCENARIOS.items():
            key = (ns, sar)
            if key not in cache:
                sr = np.random.default_rng(seed0 + 500_000 + n)
                cache[key] = [tw.simulate_season(sr, sp * (1 + 0.1 * sr.standard_normal()),
                                                 obs_days, obs_valid, rates, sar) for _ in range(ns)]
            x = features(cache[key], d_, h_, g_, np.random.default_rng(seed0 + 900_000 + n))
            rows.append({"field": n, "scenario": name, "sp_true": sp, **x})
    return pd.DataFrame(rows)


def main():
    real = json.loads((OUT / "real_validation_metrics.json").read_text())
    season = pd.read_csv(OUT / "real_season_curve.csv", index_col="date")
    obs_days = np.array([tw.dayof(d) for d in real["dates"]], float)
    obs_valid = np.array([season.valid_rice_frac.get(d, 0.2) for d in real["dates"]])
    rates = tw.detection_rates()

    # Keyed by SAR accuracy: SAR scenarios are simulated with it, so a cache built
    # at another SAR_ACC would silently give the wrong answer.
    cache = ROOT / "data" / "cache" / f"sbi_sets{tw.RUN_TAG}.pkl"
    if cache.exists():
        train, test = pd.read_pickle(cache)
    else:
        train = build(8000, 1_000_000, obs_days, obs_valid, rates)
        # The test fields are the Experiment-4 fields (same seeds), for a like-for-like comparison.
        test = build(1000, 10_000, obs_days, obs_valid, rates)
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle((train, test), cache)
    # Ablation: the app log alone, no satellite. Irrigation intervals and depths
    # still carry leakage information, because farmers re-pump once the field is dry.
    app_only = ["irr_interval", "h0", "et0", "rain"]
    APP = "A0 app only (date, hours), no satellite"

    def ablate(df):
        a = df[df.scenario == "S2 + app date & pump hours"].copy()
        a["scenario"] = APP
        a[[c for c in a.columns if c not in app_only + ["field", "scenario", "sp_true"]]] = np.nan
        return pd.concat([df, a], ignore_index=True)

    train, test = ablate(train), ablate(test)
    scenarios = list(tw.SCENARIOS) + [APP]
    cols = [c for c in train.columns if c not in ("field", "scenario", "sp_true")]

    exp4 = pd.read_csv(OUT / "twin_fields.csv.gz")
    et_mid = float(tw.wx.loc["2024-02-01":"2024-04-15"].et0.mean())
    kc = hydro.kc_rice(2.0)
    out = {}
    for name in scenarios:
        tr, te = train[train.scenario == name], test[test.scenario == name]
        use = [c for c in cols if tr[c].notna().mean() > 0.5]
        q = {}
        for a in (0.05, 0.5, 0.95):
            m = HistGradientBoostingRegressor(loss="quantile", quantile=a, max_iter=400,
                                              learning_rate=0.05, random_state=0)
            m.fit(tr[use], np.log(tr.sp_true))
            q[a] = np.exp(m.predict(te[use]))
        t = te.sp_true.to_numpy()
        D = 50 / (kc * et_mid + t)
        err = abs(50 / (kc * et_mid + q[0.5]) - D)
        err_reg = abs(50 / (kc * et_mid + 5.0) - D)
        leaky_t, leaky_h = t > tw.LEAKY, q[0.5] > tw.LEAKY
        e4 = exp4[exp4.scenario == name].dropna(subset=["sp_hat"])
        if e4.empty:
            e4 = pd.DataFrame({"sp_true": [np.nan, 1.0], "sp_hat": [np.nan, 1.0], "q05": [np.nan] * 2, "q95": [np.nan] * 2})
        out[name] = {
            "features": use,
            "sbi_median_abs_pct_error": float(np.median(abs(q[0.5] - t) / t) * 100),
            "sbi_spearman": float(spearmanr(t, q[0.5]).correlation),
            "sbi_ci90_coverage": float(((q[0.05] <= t) & (t <= q[0.95])).mean()),
            "sbi_leaky_bacc": float(0.5 * ((leaky_t & leaky_h).sum() / max(leaky_t.sum(), 1)
                                           + (~leaky_t & ~leaky_h).sum() / max((~leaky_t).sum(), 1))),
            "sbi_next_dry_err_days": float(np.median(err)),
            "regional_const_next_dry_err_days": float(np.median(err_reg)),
            "sbi_share_err_gt1d": float((err > 1).mean()),
            "regional_share_err_gt1d": float((err_reg > 1).mean()),
            "closed_form_spearman": float(spearmanr(e4.sp_true, e4.sp_hat).correlation),
            "closed_form_ci90_coverage": float(((e4.q05 <= e4.sp_true) & (e4.sp_true <= e4.q95)).mean()),
        }
    (OUT / f"sbi_metrics{tw.RUN_TAG}.json").write_text(json.dumps(out, indent=2))
    tab = pd.DataFrame({k: {m: v for m, v in d.items() if m != "features"} for k, d in out.items()}).T
    print(tab.round(3).to_string())


if __name__ == "__main__":
    main()
