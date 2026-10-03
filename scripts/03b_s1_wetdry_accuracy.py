"""Sentinel-1 wet/dry accuracy by LAI bin, against the optical retriever's labels.

Script 03 reports median VV/VH contrasts. This script turns the same S1/S2 pairs into a
classification accuracy that can replace the 85 % SAR_ACC assumption in
scripts/04_field_leakage_twin.py. Labels come from optics (P(ponded) > 0.8 wet, < 0.2 dry),
so above LAI 2 the labels themselves are weak: treat those numbers as indicative only.

Classifier: logistic regression on (VV dB, VH dB, VH-VV dB, S1-date anomaly of each), fitted
with leave-one-S1-date-out cross-validation; reported as balanced accuracy. Run at 20 m pixels
and at 60 m blocks (3 x 3 mean in linear power, which suppresses speckle).
Run scripts/02 and 03 first (uses the S2 cache).
"""

import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
s1f = __import__("03_planetary_computer_s1_fusion")
real, data_sources = s1f.real, s1f.data_sources
OUT = ROOT / "results"
BINS = [(0, 1), (1, 2), (2, 3), (3, 7)]
RNG = np.random.default_rng(0)


def block(a, k=3):
    n = a.shape[0] // k
    with np.errstate(all="ignore"):
        return np.nanmean(a[:n * k, :n * k].reshape(n, k, n, k), axis=(1, 3))


def main():
    import pickle
    s2 = data_sources.pc_items(real.BBOX_LL, real.START, real.END, mgrs_tile=real.TILE)
    s2 = [i for i in s2 if i["cloud"] is None or i["cloud"] < real.MAX_TILE_CLOUD]
    with open(OUT / "models" / "background_aware_models.pkl", "rb") as f:
        M = pickle.load(f)
    s2d = {}
    for it in s2:
        got = real.read_item(it)
        if got is None:
            continue
        R, scl = got
        X = np.moveaxis(R, 0, -1)
        ok = np.isin(scl, real.VALID_SCL) & np.isfinite(X).all(-1)
        F = real.features(X[ok])
        p = np.full(ok.shape, np.nan, np.float32)
        lai = np.full(ok.shape, np.nan, np.float32)
        p[ok] = M["clf_ponded"].predict_proba(F)[:, 1]
        lai[ok] = M["reg_lai"].predict(F)
        ix = s1f.indices(X)
        s2d[pd.Timestamp(it["date"])] = dict(p=p, lai=lai, ok=ok, ndvi=ix["ndvi"], lswi=ix["lswi"], evi=ix["evi"])
    dts = sorted(s2d)
    early = [d for d in dts if pd.Timestamp("2023-12-15") <= d <= pd.Timestamp("2024-02-29")]
    late = [d for d in dts if pd.Timestamp("2024-03-01") <= d <= pd.Timestamp("2024-04-30")]
    flood = np.zeros((real.N, real.N), bool)
    for d in early:
        v = s2d[d]
        flood |= (v["lswi"] + 0.05 >= v["evi"]) & (v["ndvi"] > 0.05) & (v["ndvi"] < 0.4) & v["ok"]
    green = np.nanmax(np.stack([np.where(s2d[d]["ok"], s2d[d]["ndvi"], np.nan) for d in late]), 0) > 0.6
    rice = flood & green

    rows = {"20m": [], "60m": []}
    for it in data_sources.pc_s1_rtc_items(real.BBOX_LL, real.START, real.END):
        d = pd.Timestamp(it["date"])
        near = min(dts, key=lambda x: abs((x - d).days))
        if abs((near - d).days) > 3 or "VV" not in it["hrefs"]:
            continue
        vv, vh = s1f.read_s1(it["hrefs"]["VV"]), s1f.read_s1(it["hrefs"]["VH"])
        v = s2d[near]
        for scale in ("20m", "60m"):
            if scale == "20m":
                P, L, VV, VH, OK = v["p"], v["lai"], vv, vh, rice & v["ok"]
            else:
                P, L, VV, VH = block(v["p"]), block(v["lai"]), block(vv), block(vh)
                OK = (block((rice & v["ok"]).astype(float)) > 0.8)
            m = OK & np.isfinite(VV) & np.isfinite(VH) & np.isfinite(P) & ((P > 0.8) | (P < 0.2))
            vvdb, vhdb = 10 * np.log10(VV[m]), 10 * np.log10(VH[m])
            df = pd.DataFrame({"date": it["date"], "wet": (P[m] > 0.8).astype(int), "lai": L[m],
                               "vv": vvdb, "vh": vhdb, "cr": vhdb - vvdb})
            # Date anomaly: removes the date-wide offset (incidence, moisture, wind).
            for c in ("vv", "vh", "cr"):
                df[c + "_anom"] = df[c] - df[c].median()
            if len(df) > 20000:
                df = df.sample(20000, random_state=0)
            rows[scale].append(df)

    res = {"n_s1_dates_paired": None, "note": "labels from optics; LAI>2 labels are weak"}
    feats = ["vv", "vh", "cr", "vv_anom", "vh_anom", "cr_anom"]
    for scale, parts in rows.items():
        df = pd.concat(parts, ignore_index=True)
        res["n_s1_dates_paired"] = int(df.date.nunique())
        out = []
        for lo, hi in BINS:
            b = df[(df.lai >= lo) & (df.lai < hi)]
            nw, nd = int(b.wet.sum()), int((1 - b.wet).sum())
            rec = {"lai_lo": lo, "lai_hi": hi, "n_wet": nw, "n_dry": nd, "n_dates_with_wet": int(b[b.wet == 1].date.nunique())}
            if nw >= 50 and nd >= 50 and b.date.nunique() >= 3:
                yt, yp = [], []
                for dte in b.date.unique():
                    tr, te = b[b.date != dte], b[b.date == dte]
                    if tr.wet.nunique() < 2 or te.empty:
                        continue
                    clf = LogisticRegression(class_weight="balanced", max_iter=500).fit(tr[feats], tr.wet)
                    yt.append(te.wet.values)
                    yp.append(clf.predict(te[feats]))
                yt, yp = np.concatenate(yt), np.concatenate(yp)
                rec["balanced_acc_lodo"] = float(balanced_accuracy_score(yt, yp))
                rec["vh_contrast_db_median"] = float(b[b.wet == 1].vh.median() - b[b.wet == 0].vh.median())
            out.append(rec)
        res[scale] = out
    (OUT / "s1_wetdry_accuracy.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
