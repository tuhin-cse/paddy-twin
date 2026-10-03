"""Experiment 4d: what does the satellite add once the farmer also reports "water gone"?

04b compares satellite + app against an app-only baseline that has irrigation
dates and pump hours but no "no water this morning" reports. That baseline is
weaker than the S3 scenario it is compared with. Here every app-only variant is
paired with the satellite variant that has exactly the same log fields, so the
difference is the satellite's contribution alone.

Uses the 04b / 04c simulation caches (run those first). None of the scenarios
here use SAR, so the SAR_ACC setting of the cache does not matter.
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
CACHE = ROOT / "data" / "cache"

S1, S2, S3 = ("S1 + app irrigation date", "S2 + app date & pump hours",
              "S3 + app date, pump hours, water-gone")
WX = ["et0", "rain"]
# name: (scenario rows to use, features; None = every feature the scenario has)
PAIRS = {
    "app: date":                   (S1, ["irr_interval"] + WX),
    "optical + app: date":         (S1, None),
    "app: date, hours":            (S2, ["irr_interval", "h0"] + WX),
    "optical + app: date, hours":  (S2, None),
    "app: date, hours, gone":      (S3, ["irr_interval", "h0", "irr_to_gone"] + WX),
    "optical + app: date, hours, gone": (S3, None),
}


def fit(train, test, scenario, feats):
    tr, te = train[train.scenario == scenario], test[test.scenario == scenario]
    cols = [c for c in train.columns if c not in ("field", "scenario", "sp_true")]
    use = feats or [c for c in cols if tr[c].notna().mean() > 0.5]
    q = {}
    for a in (0.05, 0.5, 0.95):
        m = HistGradientBoostingRegressor(loss="quantile", quantile=a, max_iter=400,
                                          learning_rate=0.05, random_state=0)
        m.fit(tr[use], np.log(tr.sp_true))
        q[a] = np.exp(m.predict(te[use]))
    return te, q


def metrics(te, q, kc, et_mid):
    t = te.sp_true.to_numpy()
    D = 50 / (kc * et_mid + t)
    err = abs(50 / (kc * et_mid + q[0.5]) - D)
    return {"spearman": float(spearmanr(t, q[0.5]).correlation),
            "median_abs_pct_error": float(np.median(abs(q[0.5] - t) / t) * 100),
            "next_dry_err_days": float(np.median(err)),
            "share_err_gt1d": float((err > 1).mean()),
            "ci90_coverage": float(((q[0.05] <= t) & (t <= q[0.95])).mean())}


def main():
    et_mid = float(tw.wx.loc["2024-02-01":"2024-04-15"].et0.mean())
    kc = hydro.kc_rice(2.0)
    train, test = pd.read_pickle(CACHE / "sbi_sets.pkl")
    out = {"baseline": {}, "topup": {}}
    for name, (scn, feats) in PAIRS.items():
        out["baseline"][name] = metrics(*fit(train, test, scn, feats), kc, et_mid)

    # One-line physics from the app alone: S&P = h0 / (irrigation -> water gone) - ETc.
    te = test[test.scenario == S3]
    sp = (te.h0 / te.irr_to_gone - kc * te.et0).to_numpy()
    ok = np.isfinite(sp)
    out["baseline"]["app closed form: h0 / t_gone - Kc ET0"] = {
        "spearman": float(spearmanr(te.sp_true[ok], sp[ok]).correlation),
        "share_fields": float(ok.mean())}

    # Farmers who top up before the field dries (04c): the satellite matters more.
    tp = CACHE / "topup_sets.pkl"
    if tp.exists():
        train, test = pd.read_pickle(tp)
        out["topup"]["app: date, hours"] = metrics(*fit(train, test, S2, ["irr_interval", "h0"] + WX), kc, et_mid)
        out["topup"]["optical + app: date, hours"] = metrics(*fit(train, test, S2, None), kc, et_mid)

    (OUT / "app_only_ablation.json").write_text(json.dumps(out, indent=2))
    for part, d in out.items():
        print(f"\n{part}")
        print(pd.DataFrame(d).T.round(3).to_string())


if __name__ == "__main__":
    main()
