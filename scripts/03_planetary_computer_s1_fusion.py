"""Experiment 3 (Planetary Computer): does Sentinel-1 carry the water signal where optics cannot?

Run scripts/02_real_s2_boro_jamalpur.py first (either source); this script
reuses its cached S2 stack and rice mask definition.

For every Sentinel-1 RTC acquisition over the same 10 km window it:
  1. reprojects VV/VH gamma0 onto the 20 m S2 grid,
  2. pairs it with the nearest clear S2 date (<= 3 days apart),
  3. compares backscatter for rice pixels the optical retriever calls ponded
     vs drained, stratified by LAI.
If the ponded/drained separation in VV or VH survives above LAI 2, SAR can
carry the AWD signal through canopy closure and through cloud, which is the
fusion step of the method.

Needs outbound access to planetarycomputer.microsoft.com and
*.blob.core.windows.net.
"""

import json
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.transform import from_origin

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from awdrt import data_sources  # noqa: E402
from awdrt.s2 import FEATURE_BANDS, indices  # noqa: E402

real = __import__("02_real_s2_boro_jamalpur")
OUT = ROOT / "results"
DST_TRANSFORM = from_origin(real.BOUNDS[0], real.BOUNDS[3], real.PX, real.PX)


def read_s1(href):
    with rasterio.open(href) as src, WarpedVRT(
        src, crs=f"EPSG:{real.EPSG}", transform=DST_TRANSFORM, width=real.N, height=real.N,
        resampling=Resampling.average,
    ) as vrt:
        a = vrt.read(1).astype(np.float32)
    return np.where(a > 0, a, np.nan)


def main():
    s2 = data_sources.pc_items(real.BBOX_LL, real.START, real.END, mgrs_tile=real.TILE)
    s2 = [i for i in s2 if i["cloud"] is None or i["cloud"] < real.MAX_TILE_CLOUD]
    with open(OUT / "models" / "background_aware_models.pkl", "rb") as f:
        M = pickle.load(f)

    # S2 per date: p_ponded, LAI, validity.
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
        ix = indices(X)
        s2d[pd.Timestamp(it["date"])] = dict(p=p, lai=lai, ok=ok, ndvi=ix["ndvi"],
                                             lswi=ix["lswi"], evi=ix["evi"])

    # Same rice mask rule as script 02.
    dts = sorted(s2d)
    early = [d for d in dts if pd.Timestamp("2023-12-15") <= d <= pd.Timestamp("2024-02-29")]
    late = [d for d in dts if pd.Timestamp("2024-03-01") <= d <= pd.Timestamp("2024-04-30")]
    flood = np.zeros((real.N, real.N), bool)
    for d in early:
        v = s2d[d]
        flood |= (v["lswi"] + 0.05 >= v["evi"]) & (v["ndvi"] > 0.05) & (v["ndvi"] < 0.4) & v["ok"]
    green = np.nanmax(np.stack([np.where(s2d[d]["ok"], s2d[d]["ndvi"], np.nan) for d in late]), 0) > 0.6
    rice = flood & green

    rows = []
    for it in data_sources.pc_s1_rtc_items(real.BBOX_LL, real.START, real.END):
        d = pd.Timestamp(it["date"])
        near = min(dts, key=lambda x: abs((x - d).days))
        if abs((near - d).days) > 3 or "VV" not in it["hrefs"]:
            continue
        vv, vh = read_s1(it["hrefs"]["VV"]), read_s1(it["hrefs"]["VH"])
        v = s2d[near]
        m = rice & v["ok"] & np.isfinite(vv) & np.isfinite(vh)
        for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 7)]:
            band = m & (v["lai"] >= lo) & (v["lai"] < hi)
            wet, dry = band & (v["p"] > 0.8), band & (v["p"] < 0.2)
            if wet.sum() < 30 or dry.sum() < 30:
                continue
            rows.append({
                "s1_date": it["date"], "s2_date": near.date().isoformat(), "orbit": it["orbit"],
                "rel_orbit": it["rel_orbit"], "lai_lo": lo, "lai_hi": hi,
                "n_wet": int(wet.sum()), "n_dry": int(dry.sum()),
                "vv_db_wet": float(10 * np.log10(np.nanmedian(vv[wet]))),
                "vv_db_dry": float(10 * np.log10(np.nanmedian(vv[dry]))),
                "vh_db_wet": float(10 * np.log10(np.nanmedian(vh[wet]))),
                "vh_db_dry": float(10 * np.log10(np.nanmedian(vh[dry]))),
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "s1_vs_optical_background.csv", index=False)
    if df.empty:
        print("no S1/S2 pairs with enough wet and dry rice pixels")
        return
    df["dvv"] = df.vv_db_wet - df.vv_db_dry
    df["dvh"] = df.vh_db_wet - df.vh_db_dry
    summary = df.groupby("lai_lo")[["dvv", "dvh", "n_wet", "n_dry"]].median()
    print(summary.to_string())
    (OUT / "s1_fusion_summary.json").write_text(json.dumps(summary.reset_index().to_dict("records"), indent=2))


if __name__ == "__main__":
    main()
