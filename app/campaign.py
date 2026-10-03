"""Campaign data for Paddy Twin's replay mode: real (or demo) observations instead of the simulator.

A campaign is a folder app/campaigns/<name>/ with these files (CSV with a header row; dates YYYY-MM-DD and
times YYYY-MM-DDTHH:MM, local time):

  campaign.json   {"label", "site" (a site file in app/sites/), "start", "end" (the season, inclusive),
                   "synthetic" (true for demo data), "source", optional "nisar_accuracy" (measured)}
  fields.csv      field_id, transplant_date
  loggers.csv     logger_id, time, level_mm        water level in the AWD pipe; negative = below the surface
  app_log.csv     field_id, date, event [, pump_hours, depth_mm]
                  event = irrigated | water_gone; an irrigation's depth is depth_mm if given, else
                  pump_hours x the tubewell's discharge / the field's area
  satellite.csv   date, sensor, field_id, wet [, lai]
                  sensor = s2 | s1 | nisar; wet = 1/0 from the Layer 1 detector on the field polygon;
                  lai (Sentinel-2) feeds the model's LAI input
  weather.csv     date, et0_mm, rain_mm [, tmean_c]  from the season start to at least 30 days after its end; the
                  daily mean temperature drives the crop's growth (thermal time); without it, ERA5 Jamalpur's
                  temperature of the same dates is used where available
  tubewells.csv   tubewell_id, discharge_m3h         (optional; measured pump discharge)
  meters.csv      tubewell_id, date, m3              (optional; metered volume pumped on that date)

Field and logger ids must match the site file. Everything is turned into per-field daily arrays on the
season's day grid (day 0 = start), and logger readings into the 00/06/12/18 h grid the model uses.
"""

import csv
import json
import math
import pathlib

import numpy as np
import pandas as pd

CAMPAIGNS_DIR = pathlib.Path(__file__).with_name("campaigns")
PER_DAY = 4                 # 00, 06, 12, 18 h (as the simulator and the model)
SNAP_H = 3.0                # a reading counts for a grid time if it is within 3 h of it
SENSOR_HOUR = {"nisar": 6, "s2": 10.5, "s1": 18}


def _rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return [{k.strip().lower(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(fh)]


def list_campaigns():
    out = []
    for d in sorted(p for p in CAMPAIGNS_DIR.glob("*") if (p / "campaign.json").is_file()):
        meta = json.loads((d / "campaign.json").read_text())
        out.append({"name": d.name, "label": meta.get("label", d.name), "site": meta.get("site"),
                    "synthetic": bool(meta.get("synthetic", False)), "start": meta.get("start"), "end": meta.get("end")})
    return out


def load(name, layout):
    """Read and validate a campaign against its site layout (sites.load_site). Returns a dict of arrays."""
    d = CAMPAIGNS_DIR / name
    if not (d / "campaign.json").is_file() or d.parent != CAMPAIGNS_DIR:
        raise KeyError(f"unknown campaign {name!r}")
    meta = json.loads((d / "campaign.json").read_text())
    start, end = pd.Timestamp(meta["start"]), pd.Timestamp(meta["end"])
    T = (end - start).days + 1
    if not 20 <= T <= 200:
        raise ValueError("the season must be 20-200 days long")
    fid = {f["id"]: i for i, f in enumerate(layout["fields"])}
    n = len(fid)
    issues = []

    def day_of(s):
        return (pd.Timestamp(s[:10]) - start).days

    # transplanting
    tp = np.full(n, np.nan)
    for r in _rows(d / "fields.csv"):
        if r["field_id"] in fid:
            tp[fid[r["field_id"]]] = day_of(r["transplant_date"])
        else:
            issues.append(f"fields.csv: unknown field {r['field_id']}")
    if np.isnan(tp).any():
        missing = [f["id"] for i, f in enumerate(layout["fields"]) if np.isnan(tp[i])]
        raise ValueError(f"fields.csv: no transplant date for {len(missing)} fields, e.g. {missing[:3]}")

    # weather (season + the forecast horizon; padded with the season's mean if short)
    wx = {day_of(r["date"]): (float(r["et0_mm"]), float(r["rain_mm"]), float(r["tmean_c"]) if r.get("tmean_c") else np.nan)
          for r in _rows(d / "weather.csv")}
    need = T + 32
    et0, rain, tmean = np.full(need, np.nan), np.full(need, np.nan), np.full(need, np.nan)
    for k, (e, r_, tm) in wx.items():
        if 0 <= k < need:
            et0[k], rain[k], tmean[k] = e, r_, tm
    if np.isnan(tmean).any():   # fill from ERA5 Jamalpur on the same dates, else the season's mean
        era = pd.read_csv(pathlib.Path(__file__).resolve().parents[1] / "results" / "era5_daily_jamalpur.csv", index_col=0, parse_dates=True).tmean
        dates = start + pd.to_timedelta(np.arange(need), unit="D")
        fill = era.reindex(dates).to_numpy()
        n_fill = int((np.isnan(tmean) & np.isfinite(fill)).sum())
        tmean = np.where(np.isnan(tmean), fill, tmean)
        if n_fill:
            issues.append(f"weather.csv: no tmean_c on {n_fill} days; ERA5 Jamalpur temperature used for the crop's growth")
        if np.isnan(tmean).any():
            issues.append(f"weather.csv: {int(np.isnan(tmean).sum())} days without temperature, set to the mean")
            tmean[np.isnan(tmean)] = np.nanmean(tmean) if np.isfinite(tmean).any() else 22.0
    if np.isnan(et0[:T]).any():
        raise ValueError(f"weather.csv: {int(np.isnan(et0[:T]).sum())} season days without weather")
    if np.isnan(et0).any():
        issues.append(f"weather.csv: {int(np.isnan(et0).sum())} days after the season padded with the season mean")
        et0[np.isnan(et0)] = np.nanmean(et0[:T])
        rain[np.isnan(rain)] = 0.0

    # loggers -> 6-hourly grid (nearest reading within 3 h)
    by_id = {lg["id"]: lg for lg in layout["loggers"]}
    grid = {}
    raw_count = {}
    for r in _rows(d / "loggers.csv"):
        lid = r["logger_id"]
        if lid not in by_id:
            issues.append(f"loggers.csv: logger {lid} is not in the site file")
            continue
        t = (pd.Timestamp(r["time"]) - start).total_seconds() / 3600.0
        try:
            v = float(r["level_mm"])
        except ValueError:
            continue
        g = grid.setdefault(lid, {})
        raw_count[lid] = raw_count.get(lid, 0) + 1
        k = int(round(t / (24 / PER_DAY)))
        if abs(t - k * 24 / PER_DAY) <= SNAP_H and 0 <= k < T * PER_DAY + 1:
            if k not in g or abs(t - k * 6) < g[k][1]:
                g[k] = (v, abs(t - k * 6))
    logger6 = {}
    for lid in by_id:
        arr = np.full(T * PER_DAY + 1, np.nan)
        for k, (v, _) in grid.get(lid, {}).items():
            arr[k] = v
        logger6[lid] = arr
        if not raw_count.get(lid):
            issues.append(f"logger {lid}: no readings")

    # farmer app log
    irr_mm = np.zeros((n, T))
    logged = np.zeros((n, T), bool)
    gone = np.zeros((n, T), bool)
    discharge = {}
    if (d / "tubewells.csv").is_file():
        tw_index = {t["id"]: i for i, t in enumerate(layout["tubewells"])}
        for r in _rows(d / "tubewells.csv"):
            if r["tubewell_id"] in tw_index:
                discharge[tw_index[r["tubewell_id"]]] = float(r["discharge_m3h"])
    for r in _rows(d / "app_log.csv"):
        if r["field_id"] not in fid:
            issues.append(f"app_log.csv: unknown field {r['field_id']}")
            continue
        f, k = fid[r["field_id"]], day_of(r["date"])
        if not 0 <= k < T:
            continue
        if r["event"] == "irrigated":
            if r.get("depth_mm"):
                dep = float(r["depth_mm"])
            elif r.get("pump_hours") and layout["fields"][f]["tw"] in discharge:
                dep = float(r["pump_hours"]) * discharge[layout["fields"][f]["tw"]] / layout["fields"][f]["area_m2"] * 1000
            else:
                dep = 0.0
                issues.append(f"app_log.csv: irrigation of {r['field_id']} on {r['date']} has no depth (no pump discharge)")
            irr_mm[f, k] += dep
            logged[f, k] = True
        elif r["event"] == "water_gone":
            gone[f, k] = True

    # satellite passes
    sat = {s: np.full((n, T), -1, np.int8) for s in SENSOR_HOUR}
    lai_s2 = np.full((n, T), np.nan)
    for r in _rows(d / "satellite.csv"):
        s = r["sensor"].lower()
        if s not in sat or r["field_id"] not in fid:
            continue
        f, k = fid[r["field_id"]], day_of(r["date"])
        if 0 <= k < T:
            sat[s][f, k] = int(float(r["wet"]) > 0.5)
            if s == "s2" and r.get("lai"):
                lai_s2[f, k] = float(r["lai"])

    meters = []
    if (d / "meters.csv").is_file():
        meters = [{"tubewell": r["tubewell_id"], "day": day_of(r["date"]), "m3": float(r["m3"])} for r in _rows(d / "meters.csv")]

    t_mid = (pd.Timestamp(f"{start.year if start.month <= 3 else start.year + 1}-03-01") - start).days
    return {"meta": meta, "name": name, "start": start, "T": T, "n": n, "tp": tp, "et0": et0, "rain": rain,
            "logger6": logger6, "irr_mm": irr_mm, "logged": logged, "gone": gone, "sat": sat, "lai_s2": lai_s2,
            "meters": meters, "t_mid": t_mid, "issues": issues, "discharge": discharge, "tmean": tmean,
            "nisar_accuracy": meta.get("nisar_accuracy")}


def daily_lai(c, lai_curve):
    """The model's daily LAI input: Sentinel-2 LAI interpolated per field, or the nominal curve from the
    transplanting date where a field has fewer than two Sentinel-2 LAI values."""
    n, T = c["n"], c["T"]
    days = np.arange(T) + 0.5
    out = np.zeros((n, T))
    for f in range(n):
        nominal = lai_curve(days, c["tp"][f], 1.0)
        ok = np.isfinite(c["lai_s2"][f])
        if ok.sum() >= 2:
            out[f] = np.interp(days, np.flatnonzero(ok) + 0.5, c["lai_s2"][f, ok])
        else:
            out[f] = nominal
        out[f, days + 0.5 < c["tp"][f]] = 0.0
    return out


def finite_or_none(x, nd=1):
    return None if x is None or not math.isfinite(x) else round(float(x), nd)
