"""Cross-check ERA5-derived FAO-56 ET0 and rain (results/era5_daily_jamalpur.csv) against
NASA POWER and Open-Meteo (ERA5-Land-based archive, its own FAO-56 ET0) for the same point and days."""

import json
import pathlib

import numpy as np
import pandas as pd
import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
LAT, LON = 25.205, 89.985


def power(start, end):
    r = requests.get("https://power.larc.nasa.gov/api/temporal/daily/point", timeout=120, params=dict(
        parameters="T2M_MAX,T2M_MIN,T2M,T2MDEW,WS2M,ALLSKY_SFC_SW_DWN,PRECTOTCORR,PS",
        community="AG", latitude=LAT, longitude=LON, start=start.replace("-", ""), end=end.replace("-", ""),
        format="JSON"))
    r.raise_for_status()
    p = pd.DataFrame(r.json()["properties"]["parameter"])
    p.index = pd.to_datetime(p.index, format="%Y%m%d")
    p = p.replace(-999.0, np.nan)
    # FAO-56 Penman-Monteith from POWER met (radiation MJ m-2 d-1, wind already at 2 m).
    tmax, tmin, tmean = p.T2M_MAX, p.T2M_MIN, (p.T2M_MAX + p.T2M_MIN) / 2
    es = lambda t: 0.6108 * np.exp(17.27 * t / (t + 237.3))  # noqa: E731
    es_ = (es(tmax) + es(tmin)) / 2
    ea = es(p.T2MDEW)
    delta = 4098 * es(tmean) / (tmean + 237.3) ** 2
    gamma = 0.000665 * p.PS
    J = p.index.dayofyear.values
    phi = np.radians(LAT)
    dr = 1 + 0.033 * np.cos(2 * np.pi * J / 365)
    dec = 0.409 * np.sin(2 * np.pi * J / 365 - 1.39)
    ws = np.arccos(-np.tan(phi) * np.tan(dec))
    ra = 24 * 60 / np.pi * 0.082 * dr * (ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))
    rs = p.ALLSKY_SFC_SW_DWN
    rso = (0.75 + 2e-5 * 20) * ra
    rnl = 4.903e-9 * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2 * (0.34 - 0.14 * np.sqrt(ea)) * \
        (1.35 * np.clip(rs / rso, 0, 1) - 0.35)
    rn = 0.77 * rs - rnl
    et0 = (0.408 * delta * rn + gamma * 900 / (tmean + 273) * p.WS2M * (es_ - ea)) / \
        (delta + gamma * (1 + 0.34 * p.WS2M))
    return pd.DataFrame({"et0_power": et0, "rain_power": p.PRECTOTCORR, "rs_power": rs})


def openmeteo(start, end):
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", timeout=120, params=dict(
        latitude=LAT, longitude=LON, start_date=start, end_date=end, timezone="Asia/Dhaka",
        daily="et0_fao_evapotranspiration,precipitation_sum,shortwave_radiation_sum"))
    r.raise_for_status()
    d = pd.DataFrame(r.json()["daily"]).set_index("time")
    d.index = pd.to_datetime(d.index)
    return d.rename(columns={"et0_fao_evapotranspiration": "et0_om", "precipitation_sum": "rain_om",
                             "shortwave_radiation_sum": "rs_om"})


def main():
    e = pd.read_csv(OUT / "era5_daily_jamalpur.csv", index_col=0, parse_dates=True)
    s, t = str(e.index.min().date()), str(e.index.max().date())
    df = e[["et0", "rain", "rs"]].join(power(s, t)).join(openmeteo(s, t))
    df.to_csv(OUT / "et0_crosscheck_daily.csv")
    res = {"period": [s, t], "n_days": int(len(df))}
    for k in ("power", "om"):
        d = (df["et0"] - df[f"et0_{k}"]).dropna()
        res[f"era5_vs_{k}"] = {
            "et0_bias_mm_d": float(d.mean()), "et0_rmse_mm_d": float(np.sqrt((d ** 2).mean())),
            "et0_r": float(df["et0"].corr(df[f"et0_{k}"])),
            "rs_bias_MJ": float((df["rs"] - df[f"rs_{k}"]).mean()),
            "rain_total_mm": [float(df["rain"].sum()), float(df[f"rain_{k}"].sum())],
        }
    m = df.groupby(df.index.month)[["et0", "et0_power", "et0_om"]].mean().round(2)
    res["monthly_mean_et0"] = {int(k): v for k, v in m.to_dict("index").items()}
    (OUT / "et0_crosscheck.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
