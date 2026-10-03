"""Daily FAO-56 reference evapotranspiration and rainfall at a point from ERA5.

Reads the NSF NCAR ERA5 mirror on AWS Open Data (ds633.0, s3://nsf-ncar-era5)
with HTTP range requests, so only the chunks around the point are fetched.
ET0 follows FAO-56 Penman-Monteith (Allen et al. 1998, eqs. 6-40) on daily
aggregates of the hourly analysis fields.
"""

import datetime as dt
import pathlib
import time

import fsspec
import h5py
import numpy as np
import pandas as pd

BUCKET = "https://nsf-ncar-era5.s3.amazonaws.com"
AN = {"2t": "128_167_2t", "2d": "128_168_2d", "10u": "128_165_10u", "10v": "128_166_10v",
      "sp": "128_134_sp"}
# No total-precipitation file in ds633.0 accumu: rain = large-scale + convective.
FC = {"ssrd": "128_169_ssrd", "lsp": "128_142_lsp", "cp": "128_143_cp"}


def _point(url, lat, lon):
    with fsspec.open(url, block_size=2 ** 20).open() as f, h5py.File(f, "r") as h:
        la, lo = h["latitude"][:], h["longitude"][:]
        i, j = int(np.argmin(abs(la - lat))), int(np.argmin(abs(lo - lon % 360)))
        var = [k for k in h.keys() if h[k].ndim >= 3][0]  # the one gridded field
        if "forecast_initial_time" in h:
            init = h["forecast_initial_time"][:]
            hours = h["forecast_hour"][:]
            vals = h[var][:, :, i, j]
            base = dt.datetime(1900, 1, 1)
            times = [base + dt.timedelta(hours=float(a) + float(b)) for a in init for b in hours]
            return pd.Series(vals.ravel(), index=pd.DatetimeIndex(times))
        t = h["time"][:]
        base = dt.datetime(1900, 1, 1)
        return pd.Series(h[var][:, i, j], index=pd.DatetimeIndex(
            [base + dt.timedelta(hours=float(x)) for x in t]))


def _retry(fn, *a, tries=5):
    for k in range(tries):
        try:
            return fn(*a)
        except Exception:  # S3 answers 503 SlowDown under load
            if k == tries - 1:
                raise
            time.sleep(2 ** (k + 1))


def _keys(prefix, month):
    import re
    import urllib.request
    xml = urllib.request.urlopen(f"{BUCKET}/?list-type=2&prefix={prefix}/{month}/", timeout=60).read().decode()
    return re.findall(r"<Key>([^<]+)</Key>", xml)


def hourly(lat, lon, start, end, cache_dir=None):
    months = pd.period_range(start, end, freq="M").strftime("%Y%m")
    cache = pathlib.Path(cache_dir) if cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
    cols = {}
    for name, code in {**AN, **FC}.items():
        prefix = "e5.oper.an.sfc" if name in AN else "e5.oper.fc.sfc.accumu"
        parts = []
        for m in months:
            for k in _retry(_keys, prefix, m):
                if f".{code}." not in k:
                    continue
                f = cache / (k.split("/")[-1] + f".{lat}_{lon}.pkl") if cache else None
                if f and f.exists():
                    parts.append(pd.read_pickle(f))
                    continue
                s = _retry(_point, f"{BUCKET}/{k}", lat, lon)
                if f:
                    s.to_pickle(f)
                parts.append(s)
        s = pd.concat(parts).sort_index()
        cols[name] = s[~s.index.duplicated()]
    return pd.DataFrame(cols)


def fao56_daily(h, lat_deg, elev_m=20.0):
    """Daily ET0 (mm/d) and rain (mm/d) from hourly ERA5 (UTC; Bangladesh = UTC+6)."""
    h = h.copy()
    h.index = h.index + pd.Timedelta(hours=6)  # local days
    T = h["2t"] - 273.15
    Td = h["2d"] - 273.15
    d = pd.DataFrame({
        "tmax": T.resample("D").max(), "tmin": T.resample("D").min(), "tmean": T.resample("D").mean(),
        "tdew": Td.resample("D").mean(),
        "u10": np.hypot(h["10u"], h["10v"]).resample("D").mean(),
        "p_kpa": (h["sp"] / 1000).resample("D").mean(),
        "rs": (h["ssrd"] / 1e6).resample("D").sum(),     # hourly accumulations, MJ/m2/d
        "rain": ((h["lsp"] + h["cp"]) * 1000).resample("D").sum(),  # m -> mm
    })
    es = lambda t: 0.6108 * np.exp(17.27 * t / (t + 237.3))  # noqa: E731
    d["es"] = (es(d.tmax) + es(d.tmin)) / 2
    d["ea"] = es(d.tdew)
    delta = 4098 * es(d.tmean) / (d.tmean + 237.3) ** 2
    gamma = 0.000665 * d.p_kpa
    u2 = d.u10 * 4.87 / np.log(67.8 * 10 - 5.42)
    J = d.index.dayofyear.to_numpy()
    phi = np.radians(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * J / 365)
    dec = 0.409 * np.sin(2 * np.pi * J / 365 - 1.39)
    ws = np.arccos(-np.tan(phi) * np.tan(dec))
    ra = 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))
    rso = (0.75 + 2e-5 * elev_m) * ra
    rns = 0.77 * d.rs
    sigma = 4.903e-9
    rnl = sigma * ((d.tmax + 273.16) ** 4 + (d.tmin + 273.16) ** 4) / 2 \
        * (0.34 - 0.14 * np.sqrt(d.ea)) * (1.35 * np.clip(d.rs / rso, 0.25, 1.0) - 0.35)
    rn = rns - rnl
    d["et0"] = (0.408 * delta * rn + gamma * 900 / (d.tmean + 273) * u2 * (d.es - d.ea)) \
        / (delta + gamma * (1 + 0.34 * u2))
    return d
