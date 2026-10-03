"""Experiment 4c: feasibility when farmers top up before the field dries.

Experiments 4/4b assume every farmer waits until the standing water is gone.
Under continuous flooding many farmers re-pump while water still stands, so the
satellite may never see a dry field and the leak cannot be timed. Here half of
the farmers top up at 5-20 mm; a leaky field still dries between top-ups, a
tight one never does. SBI is retrained on that behaviour mix.
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
sbi = __import__("04b_simulation_based_inference")
OUT = tw.OUT
SCEN = {k: v for k, v in tw.SCENARIOS.items() if k.split()[0] in ("S0", "S2", "S5", "S6")}


def build(n, seed0, obs_days, obs_valid, rates, topup_frac):
    rows = []
    for i in range(n):
        fr = np.random.default_rng(seed0 + i)
        sp = float(np.clip(fr.lognormal(np.log(5.0), 0.6), 0.5, 30))
        cache = {}
        for name, (ns, sar, d_, h_, g_) in SCEN.items():
            if (ns, sar) not in cache:
                sr = np.random.default_rng(seed0 + 500_000 + i)
                cache[(ns, sar)] = [tw.simulate_season(sr, sp * (1 + 0.1 * sr.standard_normal()), obs_days,
                                                       obs_valid, rates, sar, topup_frac) for _ in range(ns)]
            seasons = cache[(ns, sar)]
            x = sbi.features(seasons, d_, h_, g_, np.random.default_rng(seed0 + 900_000 + i))
            dried = np.mean([s["gone_t"].size > 0 for s in seasons])
            rows.append({"field": i, "scenario": name, "sp_true": sp, "ever_dried": dried, **x})
    return pd.DataFrame(rows)


def main():
    real = json.loads((OUT / "real_validation_metrics.json").read_text())
    season = pd.read_csv(OUT / "real_season_curve.csv", index_col="date")
    obs_days = np.array([tw.dayof(d) for d in real["dates"]], float)
    obs_valid = np.array([season.valid_rice_frac.get(d, 0.2) for d in real["dates"]])
    rates = tw.detection_rates()
    cache = ROOT / "data" / "cache" / f"topup_sets{tw.RUN_TAG}.pkl"  # keyed by SAR_ACC, as in 04b
    if cache.exists():
        train, test = pd.read_pickle(cache)
    else:
        train = build(4000, 3_000_000, obs_days, obs_valid, rates, 0.5)
        test = build(1000, 4_000_000, obs_days, obs_valid, rates, 0.5)
        pd.to_pickle((train, test), cache)
    cols = [c for c in train.columns if c not in ("field", "scenario", "sp_true", "ever_dried")]
    et_mid = float(tw.wx.loc["2024-02-01":"2024-04-15"].et0.mean())
    kc = tw.hydro.kc_rice(2.0)
    out = {}
    for name in SCEN:
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
        never = te.ever_dried.to_numpy() == 0
        out[name] = {
            "share_fields_never_dry": float(never.mean()),
            "spearman": float(spearmanr(t, q[0.5]).correlation),
            "median_abs_pct_error": float(np.median(abs(q[0.5] - t) / t) * 100),
            "median_abs_pct_error_never_dry": float(np.median((abs(q[0.5] - t) / t)[never]) * 100) if never.any() else None,
            "ci90_coverage": float(((q[0.05] <= t) & (t <= q[0.95])).mean()),
            "next_dry_err_days": float(np.median(err)),
            "regional_const_err_days": float(np.median(err_reg)),
            "share_err_gt1d": float((err > 1).mean()),
        }
    (OUT / f"topup_metrics{tw.RUN_TAG}.json").write_text(json.dumps(out, indent=2))
    print(pd.DataFrame(out).T.round(3).to_string())


if __name__ == "__main__":
    main()
