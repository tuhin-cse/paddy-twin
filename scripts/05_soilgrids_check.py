"""Check the real S&P map against SoilGrids 2.0 texture (sand, clay; 0-5 and 5-15 cm).

Expectation: sandier / less clayey blocks leak faster, so Spearman(S&P, sand) > 0 and
Spearman(S&P, clay) < 0. SoilGrids is 250 m, so the check is also run on S&P aggregated
to 240 m (4x4 blocks of 60 m) to avoid counting one soil pixel many times.
"""

import json
import pathlib
import warnings

import numpy as np
import rasterio
import requests
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
CACHE = ROOT / "data" / "cache" / "soilgrids"
CACHE.mkdir(parents=True, exist_ok=True)
# Grid of results/real_sp_map_60m.npy: 166 x 166 blocks of 60 m, UTM 45N, NW corner (795780, 2796040).
X0, Y0, B, NB, EPSG = 795780.0, 2796040.0, 60.0, 166, 32645
WCS = ("https://maps.isric.org/mapserv?map=/map/{p}.map&SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
       "&COVERAGEID={p}_{d}_mean&FORMAT=image/tiff&SUBSET=long(89.90,90.07)&SUBSET=lat(25.13,25.28)"
       "&SUBSETTINGCRS=http://www.opengis.net/def/crs/EPSG/0/4326"
       "&OUTPUTCRS=http://www.opengis.net/def/crs/EPSG/0/4326")


def layer(prop, depth):
    f = CACHE / f"{prop}_{depth}.tif"
    if not f.exists():
        r = requests.get(WCS.format(p=prop, d=depth), timeout=120)
        r.raise_for_status()
        f.write_bytes(r.content)
    dst = np.full((NB, NB), np.nan, np.float32)
    with rasterio.open(f) as src:
        a = src.read(1).astype(np.float32)
        if src.nodata is not None:
            a[a == src.nodata] = np.nan
        reproject(a, dst, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=from_origin(X0, Y0, B, B), dst_crs=f"EPSG:{EPSG}",
                  resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
    return dst / 10.0  # g/kg -> %


def agg(a, k=4):
    n = NB // k
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a[:n * k, :n * k].reshape(n, k, n, k), axis=(1, 3))


def main():
    sp = np.load(OUT / "real_sp_map_60m.npy")
    res = {"n_blocks_60m": int(np.isfinite(sp).sum())}
    for prop in ("sand", "clay"):
        for depth in ("0-5cm", "5-15cm"):
            s = layer(prop, depth)
            ok = np.isfinite(sp) & np.isfinite(s)
            r60 = spearmanr(sp[ok], s[ok])
            spa, sa = agg(sp), agg(s)
            ok2 = np.isfinite(spa) & np.isfinite(sa)
            r240 = spearmanr(spa[ok2], sa[ok2])
            res[f"{prop}_{depth}"] = {
                "range_pct": [float(np.nanmin(s[ok])), float(np.nanmax(s[ok]))],
                "spearman_60m": float(r60.correlation), "p_60m": float(r60.pvalue), "n_60m": int(ok.sum()),
                "spearman_240m": float(r240.correlation), "p_240m": float(r240.pvalue), "n_240m": int(ok2.sum()),
            }
    (OUT / "soilgrids_check.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
