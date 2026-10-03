"""Experiment 4e: two tests of what the satellite is for.

04d showed that the satellite and the farmer's "water gone" report carry the same
information (when the ponding ended), so satellite + app barely beats a diligent
farmer. Two ways the satellite could still earn its place:

A. Revisit. With a daily sensor the end of ponding is seen to within about a day.
   The same fields are observed on:
     - the real Sentinel-2 calendar of the Jamalpur window (as in 04b);
     - a 2.5-day series with Sentinel-2-quality detection (HLS / GF-1 + HLS);
     - a daily series with visible/NIR-only detection (PlanetScope-like, no SWIR);
     - a daily series with Sentinel-2-quality detection (upper bound).
   Error rates per LAI bin come from classifiers trained on the Experiment 1
   PROSAIL spectra, with and without the SWIR bands. Optical is used below LAI 2
   only, as elsewhere; no SAR here.

B. Sparse loggers. A district of fields whose log S&P is partly predictable from
   field attributes (a stand-in for satellite embeddings, soil and terrain).
   Only a fraction of farmers log. For the fields that do not log, compare:
     M0 regional constant (median of the logged fields' estimates);
     M1 satellite-only inversion;
     M2 leak map: attributes -> S&P, trained on the logged fields' estimates;
     M3 leak map + the field's own satellite timing.
   The strength of the attribute link (R2) is unknown in reality, so it is swept
   from 0 (attributes useless) to 0.75.

This is a two-stage stand-in for the thesis method (a differentiable water balance
trained end to end); it measures the information available, not the final model.
"""

import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
tw = __import__("04_field_leakage_twin")
sb = __import__("04b_simulation_based_inference")
hydro = tw.hydro
OUT = tw.OUT
CACHE = ROOT / "data" / "cache" / "revisit_sets.pkl"

N_TRAIN, N_TEST, N_DISTRICT = 6000, 1000, 3000
CLEAR = 0.88  # mean clear-sky share of rice pixels on real Boro S2 dates
ALL_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12"]
VNIR = ["B02", "B03", "B04", "B05", "B8A"]  # PlanetScope SuperDove has no SWIR
LOGS = {"sat only": (False, False), "sat + date": (True, False), "sat + date, hours": (True, True)}
SAT_FEATS = ["n_obs", "frac_wet", "trans_per_day", "wet_run", "dry_run", "gap", "et0", "rain"]
ET_MID = float(tw.wx.loc["2024-02-01":"2024-04-15"].et0.mean())
KC = hydro.kc_rice(2.0)


def rates_from_spectra(bands, noise=0.0, seed=0):
    """Ponded-water recall and specificity per LAI bin, 5-fold CV on Experiment 1 spectra."""
    d = pd.read_csv(OUT / "sim_test_spectra.csv.gz")
    X = d[bands].to_numpy()
    if noise:
        X = X * (1 + noise * np.random.default_rng(seed).standard_normal(X.shape))
    y = (d.background == "ponded").to_numpy()
    p = cross_val_predict(HistGradientBoostingClassifier(max_iter=300, random_state=0), X, y, cv=5)
    lo = np.array(json.loads((OUT / "simulation_metrics.json").read_text())["E1_background"]["lai_bin_lo"])
    b = np.clip(np.searchsorted(lo, d.lai.to_numpy(), side="right") - 1, 0, len(lo) - 1)
    rec = np.array([(p[(b == i) & y]).mean() if ((b == i) & y).any() else 0.5 for i in range(len(lo))])
    spec = np.array([(~p[(b == i) & ~y]).mean() if ((b == i) & ~y).any() else 0.5 for i in range(len(lo))])
    return lo, rec, spec


def calendar(step):
    days = np.arange(tw.dayof("2024-01-01"), tw.dayof(tw.SEASON_END), step, dtype=float)
    return days, np.full(days.size, CLEAR)


def real_calendar():
    real = json.loads((OUT / "real_validation_metrics.json").read_text())
    season = pd.read_csv(OUT / "real_season_curve.csv", index_col="date")
    days = np.array([tw.dayof(d) for d in real["dates"]], float)
    return days, np.array([season.valid_rice_frac.get(d, 0.2) for d in real["dates"]])


def simulate(sp, seed, cal, rates):
    rng = np.random.default_rng(seed)
    return tw.simulate_season(rng, sp * (1 + 0.1 * rng.standard_normal()), *cal, rates, False)


def feats(season, seed):
    """Satellite features, the three app-log variants and the app-only features for one season."""
    row = {}
    for name, (d_, h_) in LOGS.items():
        x = sb.features([season], d_, h_, False, np.random.default_rng(seed))
        row[name] = x
    row["gone"] = sb.features([season], True, True, True, np.random.default_rng(seed))
    return row


def prior_sp(rng):
    return float(np.clip(rng.lognormal(np.log(5.0), 0.6), 0.5, 30))


def next_dry(t, est):
    D = 50 / (KC * ET_MID + t)
    return np.abs(50 / (KC * ET_MID + est) - D)


def score(t, est):
    err = next_dry(t, est)
    return {"spearman": float(spearmanr(t, est).correlation),
            "median_abs_pct_error": float(np.median(abs(est - t) / t) * 100),
            "next_dry_err_days": float(np.median(err)), "share_err_gt1d": float((err > 1).mean())}


def frame(rows, key, cols):
    return pd.DataFrame([{c: r[key].get(c, np.nan) for c in cols} for r in rows])


def gbm(X, y, small=False):
    kw = dict(max_iter=150, min_samples_leaf=10) if small else dict(max_iter=400)
    return HistGradientBoostingRegressor(learning_rate=0.05, random_state=0, **kw).fit(X, np.log(y))


def build_sets():
    s2 = tw.detection_rates()
    s2_cv, vnir = rates_from_spectra(ALL_BANDS), rates_from_spectra(VNIR)
    cals = {
        "S2, real calendar":                 (real_calendar(), s2),
        "2.5-day, S2-quality (HLS/GF-1)":    (calendar(2.5), s2),
        "daily, VNIR only (PlanetScope-like)": (calendar(1.0), vnir),
        "daily, S2-quality (upper bound)":   (calendar(1.0), s2),
    }
    sets = {"rates": {"s2_cv_bacc": ((s2_cv[1] + s2_cv[2]) / 2).tolist(),
                      "vnir_bacc": ((vnir[1] + vnir[2]) / 2).tolist(),
                      "vnir_recall": vnir[1].tolist(), "lai_bin_lo": vnir[0].tolist()},
            "fields": {}, "district": {}}
    sps = {"train": [prior_sp(np.random.default_rng(1_000_000 + n)) for n in range(N_TRAIN)],
           "test": [prior_sp(np.random.default_rng(10_000 + n)) for n in range(N_TEST)]}
    for cname, (cal, rates) in cals.items():
        print("simulating", cname, flush=True)
        sets["fields"][cname] = {
            part: [dict(sp=sp, **feats(simulate(sp, 20_000 + n + (0 if part == "test" else 5_000_000), cal, rates),
                                       900_000 + n)) for n, sp in enumerate(sps[part])]
            for part in ("train", "test")}

    # District fields for B: attributes x, log S&P = mu + sigma * (sqrt(R2) g(x) + sqrt(1-R2) eps).
    rng = np.random.default_rng(7)
    X = rng.standard_normal((N_DISTRICT, 16))
    w = rng.standard_normal(16)
    g = np.tanh(X) @ w + 0.5 * X[:, 0] * X[:, 1]
    g = (g - g.mean()) / g.std()
    eps = rng.standard_normal(N_DISTRICT)
    for cname in ("S2, real calendar", "daily, VNIR only (PlanetScope-like)"):
        cal, rates = cals[cname]
        for r2 in (0.0, 0.25, 0.5, 0.75):
            print("district", cname, r2, flush=True)
            sp = np.clip(np.exp(np.log(5.0) + 0.6 * (np.sqrt(r2) * g + np.sqrt(1 - r2) * eps)), 0.5, 30)
            sets["district"][(cname, r2)] = {
                "X": X, "sp": sp,
                "rows": [feats(simulate(s, 40_000_000 + n, cal, rates), 950_000 + n) for n, s in enumerate(sp)]}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(sets, CACHE)
    return sets


def test_a(sets):
    out = {}
    for cname, parts in sets["fields"].items():
        tr, te = parts["train"], parts["test"]
        ytr, yte = np.array([r["sp"] for r in tr]), np.array([r["sp"] for r in te])
        res = {}
        cols = SAT_FEATS + ["irr_to_dry", "irr_to_lastwet", "irr_interval", "h0"]
        for key in LOGS:
            A, B = frame(tr, key, cols), frame(te, key, cols)
            use = [c for c in cols if A[c].notna().mean() > 0.5]
            res[key] = score(yte, np.exp(gbm(A[use], ytr).predict(B[use])))
        out[cname] = res
    # App-only references (no satellite; identical for every calendar).
    tr, te = sets["fields"]["S2, real calendar"]["train"], sets["fields"]["S2, real calendar"]["test"]
    ytr, yte = np.array([r["sp"] for r in tr]), np.array([r["sp"] for r in te])
    for name, cols in {"app only: date, hours": ["irr_interval", "h0", "et0", "rain"],
                       "app only: date, hours, gone": ["irr_interval", "h0", "irr_to_gone", "et0", "rain"]}.items():
        A, B = frame(tr, "gone", cols), frame(te, "gone", cols)
        out[name] = score(yte, np.exp(gbm(A, ytr).predict(B)))
    return out


def test_b(sets):
    out = {}
    for cname in ("S2, real calendar", "daily, VNIR only (PlanetScope-like)"):
        tr = sets["fields"][cname]["train"]
        ytr = np.array([r["sp"] for r in tr])
        sat_cols = SAT_FEATS
        log_cols = SAT_FEATS + ["irr_to_dry", "irr_to_lastwet", "irr_interval", "h0"]
        m_sat = gbm(frame(tr, "sat only", sat_cols), ytr)
        m_log = gbm(frame(tr, "sat + date, hours", log_cols), ytr)
        for r2 in (0.0, 0.25, 0.5, 0.75):
            dist = sets["district"][(cname, r2)]
            X, sp = dist["X"], dist["sp"]
            S = frame(dist["rows"], "sat only", sat_cols)
            est_sat = np.exp(m_sat.predict(S))                                           # M1
            est_log = np.exp(m_log.predict(frame(dist["rows"], "sat + date, hours", log_cols)))
            for frac in (0.05, 0.1, 0.2, 0.5):
                acc = {m: [] for m in ("M0 regional constant", "M1 satellite only",
                                       "M2 leak map (attributes)", "M3 leak map + satellite")}
                for rep in range(5):
                    rng = np.random.default_rng(100 * rep + int(frac * 100))
                    logged = rng.random(sp.size) < frac
                    lab = est_log[logged]  # the logged fields' own sat + log estimates, not the truth
                    un = ~logged
                    acc["M0 regional constant"].append(np.full(un.sum(), np.median(lab)))
                    acc["M1 satellite only"].append(est_sat[un])
                    acc["M2 leak map (attributes)"].append(np.exp(gbm(X[logged], lab, True).predict(X[un])))
                    XS = np.column_stack([X, S.to_numpy()])
                    acc["M3 leak map + satellite"].append(np.exp(gbm(XS[logged], lab, True).predict(XS[un])))
                    acc.setdefault("_truth", []).append(sp[un])
                t = np.concatenate(acc.pop("_truth"))
                out[f"{cname} | R2={r2} | logged={frac:.0%}"] = {
                    m: score(t, np.concatenate(v)) for m, v in acc.items()}
                out[f"{cname} | R2={r2} | logged={frac:.0%}"]["logged fields (sat + log)"] = score(sp[logged], est_log[logged])
    return out


def main():
    sets = pd.read_pickle(CACHE) if CACHE.exists() else build_sets()
    a, b = test_a(sets), test_b(sets)
    (OUT / "revisit_sparse_loggers.json").write_text(json.dumps(
        {"detection": sets["rates"], "A_revisit": a, "B_sparse_loggers": b}, indent=2))
    lo = sets["rates"]["lai_bin_lo"]
    print("\nDetection bacc by LAI bin (CV on E1 spectra)")
    print(pd.DataFrame({"lai_lo": lo, "S2 all bands": sets["rates"]["s2_cv_bacc"],
                        "VNIR only": sets["rates"]["vnir_bacc"]}).round(3).head(6).to_string(index=False))
    print("\nA. Revisit (1,000 test fields)")
    rows = []
    for k, v in a.items():
        if "spearman" in v:
            rows.append({"calendar": k, "logs": "-", **v})
        else:
            rows += [{"calendar": k, "logs": lk, **lv} for lk, lv in v.items()]
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("\nB. Sparse loggers: unlogged fields")
    rows = [{"case": k, "method": m, **v} for k, d in b.items() for m, v in d.items()]
    print(pd.DataFrame(rows)[["case", "method", "spearman", "next_dry_err_days", "share_err_gt1d"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
