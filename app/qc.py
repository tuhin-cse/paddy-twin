"""Quality control for campaign data (Paddy Twin replay mode): what is wrong with the data, and where.

Checks, each turned into an issue with a severity, the logger or field, the days it covers and what to do:

  loggers   no readings; stopped; gaps of a day or more; readings outside the pipe's range; flat lines (a stuck
            sensor); single-reading spikes; drift (the logger reads increasingly higher or lower than the model
            expects); satellite calls that often disagree with the logger
  farmers   rises in a logger field that the farmer did not log; farmers who stopped logging while the satellites
            or the logger still see irrigations; farmers who never used the app
  network   spells without any satellite call; days when many loggers are silent; weekly metered volumes against
            the logged irrigations of the tubewell's fields

The drift check needs a reference the logger does not feed: the model's daily belief (median water level),
which uses satellites, farmer logs and weather only. A drifting logger moves away from it at a steady rate
from some day on; a hinge (flat, then a straight line) is fitted to logger minus model, robustly, over every
possible onset, and the best fit is flagged when its slope is both large in total and clear of the noise.
The model is not the truth, so a drift issue asks for a ruler reading at the pipe, not a correction.

Everything works on the season's day grid (app/campaign.py): day 0 = start, logger readings on the 6-hourly grid.
"""

import csv
import io

import numpy as np
import pandas as pd

PER_DAY = 4
SEVERITY = ("high", "medium", "low", "info")
SENSOR_NAME = {"nisar": "NISAR", "s2": "Sentinel-2", "s1": "Sentinel-1"}

GAP_MIN = 4              # grid readings (24 h) missing in a row = a gap
STOP_MARGIN = 12         # last reading more than 3 days before the season's end = stopped
FLAT_MIN = 8             # identical readings in a row (48 h) = a stuck sensor
SPIKE_MM = 60.0          # a reading this far from both neighbours, which agree with each other
DRIFT_T, DRIFT_MM = 6.0, 40.0        # drift: slope t-value and total offset by the season's end
DRIFT_MIN_BEFORE, DRIFT_MIN_AFTER = 10, 21
RISE_MM = 40.0           # an end-of-day rise this large is an irrigation (or heavy rain)
RAIN_MM = 10.0
STOP_LOG_DAYS = 14       # no app event for this long while irrigation is still seen = stopped logging
BLIND_DAYS = 5           # this many days in a row without any satellite call
OUTAGE_SHARE = 0.6       # fewer than this share of the loggers reporting on a day = network outage
METER_LOW, METER_HIGH, METER_MIN_M3 = 0.3, 1.3, 200.0
UNLOGGED_MIN = 3          # unlogged rises in one field before it is an issue


def _runs(mask):
    """(start, end) index pairs, inclusive, of the True runs in a 1-D boolean array."""
    m = np.r_[False, np.asarray(mask, bool), False]
    edges = np.flatnonzero(m[1:] != m[:-1])
    return list(zip(edges[::2], edges[1::2] - 1))


def hinge_drift(d, r, min_before=DRIFT_MIN_BEFORE, min_after=DRIFT_MIN_AFTER):
    """Fit r = a + b * max(0, d - d0) robustly (Huber IRLS) for every onset d0; return the best fit
    {onset, slope (per day), t, offset (slope x days after onset)} or None when there are too few points."""
    d, r = np.asarray(d, float), np.asarray(r, float)
    if len(d) < min_before + min_after:
        return None
    best = None
    for d0 in range(int(d[0]) + min_before, int(d[-1]) - min_after + 1):
        h = np.clip(d - d0, 0, None)
        X = np.c_[np.ones_like(d), h]
        w = np.ones_like(r)
        for _ in range(8):
            sw = np.sqrt(w)
            beta = np.linalg.lstsq(X * sw[:, None], r * sw, rcond=None)[0]
            e = r - X @ beta
            s = 1.4826 * np.median(np.abs(e)) + 1e-6
            w = np.minimum(1.0, 1.345 * s / np.maximum(np.abs(e), 1e-9))
        loss = float(np.sum(w * e ** 2))
        if best is None or loss < best[0]:
            cov = np.linalg.pinv((X * w[:, None]).T @ X)
            se = s * np.sqrt(max(cov[1, 1], 1e-12))
            best = (loss, d0, float(beta[1]), float(beta[1] / se))
    if best is None:
        return None
    _, d0, b, t = best
    return {"onset": int(d0), "slope": round(b, 2), "t": round(t, 1), "offset": round(b * (d[-1] - d0), 0)}


def check(c, lay, season, q50, passes, pipe=(-250.0, 150.0)):
    """Run every check on a loaded campaign (campaign.load) and the model's daily belief median q50 (n, T).
    Returns {"issues", "loggers", "fields", "days", "weeks", "counts"}."""
    n, T, start = c["n"], c["T"], c["start"]
    date = lambda k: (start + pd.Timedelta(days=int(k))).strftime("%Y-%m-%d")  # noqa: E731
    fids = [f_["id"] for f_ in lay["fields"]]
    issues = []

    def add(sev, kind, title, detail, action, d0, d1, logger=None, field=None):
        issues.append({"severity": sev, "kind": kind, "logger": logger, "field": None if field is None else int(field),
                       "field_id": None if field is None else fids[field], "start": int(d0), "end": int(d1),
                       "start_date": date(d0), "end_date": date(d1), "title": title, "detail": detail, "action": action})

    in_season_days = season.any(0)
    last_day = int(np.flatnonzero(in_season_days)[-1]) if in_season_days.any() else T - 1
    rain = c["rain"][:T]

    # ── loggers ───────────────────────────────────────────────────────────────
    by_pass = {}
    for p in passes:
        if p["truth"] is not None:
            by_pass.setdefault(p["f"], []).append(p)
    net_wrong = [p["wet"] != p["truth"] for ps_ in by_pass.values() for p in ps_]
    net_rate = float(np.mean(net_wrong)) if net_wrong else 0.0

    loggers, reporting = [], np.zeros(T)
    expected = np.zeros(T)
    for lg in lay["loggers"]:
        lid, f = lg["id"], lg["field"]
        arr = c["logger6"][lid][:T * PER_DAY]
        grid_in = np.repeat(season[f], PER_DAY)
        fin = np.isfinite(arr)
        got_day = fin.reshape(T, PER_DAY).any(1)
        due = season[f] & np.r_[False, season[f][:-1]]      # from the day after transplanting
        reporting += got_day & due
        expected += due
        summ = {"id": lid, "field": f, "field_id": fids[f], "coverage": round(float((fin & grid_in).sum() / max(1, grid_in.sum())), 3),
                "first_day": None, "last_day": None, "drift": None, "sat_checked": 0, "sat_wrong": 0}
        loggers.append(summ)
        if not (fin & grid_in).any():
            add("high", "no_data", f"Logger {lid} sent no readings", "No reading in the season.",
                "Check the unit, its battery and the upload.", 0, last_day, lid, f)
            continue
        idx = np.flatnonzero(fin & grid_in)
        k_first, k_last = int(idx[0]), int(idx[-1])
        summ["first_day"], summ["last_day"] = k_first // PER_DAY, k_last // PER_DAY
        season_k = np.flatnonzero(grid_in)
        if k_last < season_k[-1] - STOP_MARGIN:
            add("high", "stopped", f"Logger {lid} stopped", f"Last reading {date(k_last // PER_DAY)}; "
                f"{(season_k[-1] - k_last) / PER_DAY:.0f} days without data since.",
                "Visit: battery, memory or a broken cable; read out the unit.", k_last // PER_DAY, last_day, lid, f)
        miss = grid_in & ~fin
        miss[:k_first] = False
        miss[k_last + 1:] = False
        for a, b in _runs(miss):
            if b - a + 1 >= GAP_MIN:
                add("medium", "gap", f"Logger {lid}: gap of {(b - a + 1) / PER_DAY:.1f} days",
                    f"No readings from {date(a // PER_DAY)} to {date(b // PER_DAY)}.",
                    "Read out the unit (the data may be on it); check the upload schedule.", a // PER_DAY, b // PER_DAY, lid, f)
        if k_first > season_k[0] + 2 * PER_DAY:
            add("low", "late_start", f"Logger {lid} started late", f"First reading {date(k_first // PER_DAY)}, "
                f"{(k_first - season_k[0]) / PER_DAY:.0f} days after transplanting.", "None now; note it for the next season.",
                season_k[0] // PER_DAY, k_first // PER_DAY, lid, f)
        lo, hi = pipe
        bad = fin & ((arr < lo - 5) | (arr > hi + 5))
        if bad.any():
            k = np.flatnonzero(bad)
            add("medium", "out_of_range", f"Logger {lid}: {len(k)} impossible readings",
                f"Outside the pipe's range ({lo:.0f} to {hi:.0f} mm), e.g. {arr[k[0]]:.0f} mm on {date(k[0] // PER_DAY)}.",
                "Check the logger's zero and the pipe's installation depth.", k[0] // PER_DAY, k[-1] // PER_DAY, lid, f)
        v = arr.copy()
        same = np.r_[False, np.abs(np.diff(v)) < 0.05] & fin & np.r_[False, fin[:-1]] & (v > lo + 1) & (v < hi - 1)
        for a, b in _runs(same):
            if b - a + 2 >= FLAT_MIN:
                add("medium", "flatline", f"Logger {lid}: stuck at {v[a]:.0f} mm",
                    f"The same reading {b - a + 2} times in a row, {date((a - 1) // PER_DAY)} to {date(b // PER_DAY)}.",
                    "A stuck float or sensor: clean the pipe, test the logger in a bucket.", (a - 1) // PER_DAY, b // PER_DAY, lid, f)
        prev, nxt = np.r_[np.nan, v[:-1]], np.r_[v[1:], np.nan]
        with np.errstate(invalid="ignore"):
            spike = (np.abs(v - prev) > SPIKE_MM) & (np.abs(v - nxt) > SPIKE_MM) & (np.sign(v - prev) == np.sign(v - nxt)) \
                & (np.abs(prev - nxt) < SPIKE_MM / 2)
        if spike.any():
            k = np.flatnonzero(spike)
            add("low", "spike", f"Logger {lid}: {len(k)} single-reading spikes",
                f"Readings that jump and come straight back, first on {date(k[0] // PER_DAY)}.",
                "Usually electrical noise; ignored by the model input, worth a cable check.", k[0] // PER_DAY, k[-1] // PER_DAY, lid, f)

        # drift against the model's belief
        daily = c["logger6"][lid][PER_DAY::PER_DAY][:T]
        ok = season[f] & np.isfinite(daily) & np.isfinite(q50[f]) & (daily > lo + 1) & (daily < hi - 1)
        dr = hinge_drift(np.flatnonzero(ok), (daily - q50[f])[ok])
        summ["drift"] = dr
        if dr and abs(dr["t"]) >= DRIFT_T and abs(dr["offset"]) >= DRIFT_MM:
            way = "higher" if dr["slope"] > 0 else "lower"
            add("medium", "drift", f"Logger {lid} drifts {way} ({dr['slope']:+.1f} mm/day)",
                f"From about {date(dr['onset'])} it reads increasingly {way} than the model expects, "
                f"{dr['offset']:+.0f} mm by {date(int(np.flatnonzero(ok)[-1]))} (t = {dr['t']:.0f}).",
                "Take a ruler reading at the pipe and compare; re-zero or replace the logger.",
                dr["onset"], int(np.flatnonzero(ok)[-1]), lid, f)

        ps_ = by_pass.get(f, [])
        summ["sat_checked"], summ["sat_wrong"] = len(ps_), int(sum(p["wet"] != p["truth"] for p in ps_))
        if len(ps_) >= 6 and summ["sat_wrong"] / len(ps_) >= max(0.5, 2 * net_rate):
            add("low", "sat_disagree", f"Satellites often disagree with logger {lid}",
                f"{summ['sat_wrong']} of {len(ps_)} wet/dry calls disagree (network: {100 * net_rate:.0f} %).",
                "Check the pipe sits in a typical part of the field and the field polygon is right.",
                min(p["d"] for p in ps_), max(p["d"] for p in ps_), lid, f)

    for s in loggers:
        sev = [i["severity"] for i in issues if i["logger"] == s["id"]]
        s["status"] = "bad" if "high" in sev else "warn" if "medium" in sev else "note" if sev else "ok"
        s["issues"] = len(sev)

    # ── farmers ───────────────────────────────────────────────────────────────
    has_app = c["logged"].any(1) | c["gone"].any(1)
    L_day = np.full((n, T), np.nan)
    for lg in lay["loggers"]:
        L_day[lg["field"]] = np.where(np.isfinite(L_day[lg["field"]]), L_day[lg["field"]],
                                      c["logger6"][lg["id"]][PER_DAY::PER_DAY][:T])
    drifting = {i["field"] for i in issues if i["kind"] in ("drift", "flatline", "spike")}
    rises = np.zeros((n, T), bool)
    with np.errstate(invalid="ignore"):
        rises[:, 1:] = (np.diff(L_day, axis=1) > RISE_MM) & season[:, 1:] & (rain[None, 1:] < RAIN_MM)
    logged_near = c["logged"] | np.c_[np.zeros((n, 1), bool), c["logged"][:, :-1]] | np.c_[c["logged"][:, 1:], np.zeros((n, 1), bool)]
    unlogged = rises & ~logged_near
    sat_seq = []
    for s_, hr in (("nisar", 6), ("s2", 10.5), ("s1", 18)):
        for f_, d_ in zip(*np.nonzero((c["sat"][s_] >= 0) & season)):
            sat_seq.append((f_, d_ + hr / 24, c["sat"][s_][f_, d_]))
    sat_seq.sort()
    rewet = np.zeros((n, T), bool)
    for i in range(1, len(sat_seq)):
        (f0, t0, w0), (f1, t1, w1) = sat_seq[i - 1], sat_seq[i]
        if f0 == f1 and w0 == 0 and w1 == 1 and rain[int(t1)] < RAIN_MM:
            rewet[f1, int(t1)] = True
    fields = []
    for f in range(n):
        ev = np.flatnonzero(c["logged"][f] | c["gone"][f])
        fields.append({"field": f, "field_id": fids[f], "app": bool(has_app[f]), "log_events": int(len(ev)),
                       "last_log_day": int(ev[-1]) if len(ev) else None, "unlogged_rises": int(unlogged[f].sum())})
        if not has_app[f] or f in drifting:
            continue
        if unlogged[f].sum() >= UNLOGGED_MIN:
            k = np.flatnonzero(unlogged[f])
            add("low", "unlogged", f"Field {fids[f]}: {len(k)} irrigations not logged",
                "The logger shows a rise of more than 40 mm, without rain, on " + ", ".join(date(x) for x in k[:5])
                + (" …" if len(k) > 5 else "") + "; the farmer logged nothing that day.",
                "Remind the farmer; check the app's reminder works.", k[0], k[-1], field=f)
        if len(ev) >= 3:
            after = ev[-1] + 1
            seen = int(rewet[f, after:].sum() + rises[f, after:].sum())
            quiet = last_day - ev[-1]
            if quiet >= STOP_LOG_DAYS and seen:
                add("low", "stopped_logging", f"Field {fids[f]}: farmer stopped logging",
                    f"Last app entry {date(ev[-1])} ({quiet} days ago); {seen} re-wetting(s) seen since by satellite or logger.",
                    "Call the farmer: phone, app or a lost login.", ev[-1], last_day, field=f)
    no_app = [fids[f] for f in range(n) if not has_app[f]]
    if no_app:
        add("info", "no_app", f"{len(no_app)} of {n} farmers never used the app",
            "No irrigation or water-gone entry all season: " + ", ".join(no_app[:12]) + (" …" if len(no_app) > 12 else "") + ".",
            "Their fields rely on satellites only; a visit or a phone call may help.", 0, last_day)

    # ── network ───────────────────────────────────────────────────────────────
    calls = np.zeros(T, int)
    by_sensor = {}
    for s_, a in c["sat"].items():
        k = ((a >= 0) & season).sum(0)
        by_sensor[s_] = k
        calls += k
    for a, b in _runs((calls == 0) & in_season_days):
        if b - a + 1 >= BLIND_DAYS:
            add("low", "blind", f"No satellite data for {b - a + 1} days", f"No usable pass on any field from {date(a)} to {date(b)}.",
                "Expect wider model ranges; farmer answers and loggers carry the forecast.", a, b)
    share = np.where(expected > 0, reporting / np.maximum(expected, 1), np.nan)
    with np.errstate(invalid="ignore"):
        low = (share < OUTAGE_SHARE) & (expected >= 5)
    for a, b in _runs(low):
        add("medium", "outage", f"Many loggers silent for {b - a + 1} day(s)",
            f"Fewer than {OUTAGE_SHARE:.0%} of the loggers reported from {date(a)} to {date(b)} (lowest {np.nanmin(share[a:b + 1]):.0%}).",
            "A network or upload problem rather than single units: check the gateway.", a, b)

    meters = []
    areas = np.array([f_["area_m2"] for f_ in lay["fields"]])
    tw_index = {t["id"]: i for i, t in enumerate(lay["tubewells"])}
    for m in c["meters"]:
        t_i = tw_index.get(m["tubewell"])
        if t_i is None or not 0 <= m["day"] < T:
            continue
        fs = [f for f in range(n) if lay["fields"][f]["tw"] == t_i]
        d0 = max(0, m["day"] - 6)
        logged_m3 = float(sum(c["irr_mm"][f, d0:m["day"] + 1].sum() * areas[f] / 1000 for f in fs))
        ratio = logged_m3 / m["m3"] if m["m3"] > 0 else None
        meters.append({"tubewell": m["tubewell"], "day": m["day"], "date": date(m["day"]), "metered_m3": round(m["m3"], 1),
                       "logged_m3": round(logged_m3, 1), "ratio": None if ratio is None else round(ratio, 2)})
        if m["m3"] >= METER_MIN_M3 and ratio is not None and (ratio < METER_LOW or ratio > METER_HIGH):
            hi_ = ratio > METER_HIGH
            add("medium" if hi_ else "low", "meter", f"Tubewell {m['tubewell']}: logged {ratio:.0%} of the metered water",
                f"Week to {date(m['day'])}: metered {m['m3']:.0f} m³, farmers logged {logged_m3:.0f} m³.",
                "Logged more than pumped: check the pump discharge and the meter." if hi_ else
                "Most pumping is not logged: remind the farmers on this tubewell.", d0, m["day"])

    # ── per day and per week ──────────────────────────────────────────────────
    days = {"loggers_reporting": reporting.astype(int).tolist(), "loggers_expected": expected.astype(int).tolist(),
            "satellite_calls": {k: v.astype(int).tolist() for k, v in by_sensor.items()},
            "logged_irrigations": c["logged"].sum(0).astype(int).tolist()}
    monday = start - pd.Timedelta(days=start.weekday())
    weeks = []
    for w0 in range(-(start - monday).days, T, 7):
        a, b = max(0, w0), min(T - 1, w0 + 6)
        if a > b:
            continue
        iso = (start + pd.Timedelta(days=a)).isocalendar()
        new = [i for i in issues if a <= i["start"] <= b and i["severity"] != "info"]
        wm = [m for m in meters if a <= m["day"] <= b]
        weeks.append({"week": f"{iso[0]}-W{iso[1]:02d}", "start": a, "end": b, "start_date": date(a), "end_date": date(b),
                      "logger_uptime": round(float(reporting[a:b + 1].sum() / max(1, expected[a:b + 1].sum())), 3),
                      "satellite_calls": {k: int(v[a:b + 1].sum()) for k, v in by_sensor.items()},
                      "logged_irrigations": int(c["logged"][:, a:b + 1].sum()),
                      "farmers_logging": int((c["logged"][:, a:b + 1] | c["gone"][:, a:b + 1]).any(1).sum()),
                      "metered_m3": round(sum(m["metered_m3"] for m in wm), 0), "logged_m3": round(sum(m["logged_m3"] for m in wm), 0),
                      "new_issues": [i["title"] for i in new],
                      "open_high": sum(1 for i in issues if i["severity"] == "high" and i["start"] <= b and i["end"] >= a)})

    order = {s: i for i, s in enumerate(SEVERITY)}
    issues.sort(key=lambda i: (order[i["severity"]], i["start"], i["logger"] or "", i["field"] or -1))
    for k, i in enumerate(issues):
        i["id"] = k
    counts = {s: sum(1 for i in issues if i["severity"] == s) for s in SEVERITY}
    return {"issues": issues, "loggers": loggers, "fields": fields, "days": days, "weeks": weeks, "meters": meters,
            "counts": counts, "network_sat_disagree": round(net_rate, 3)}


def issues_csv(qc):
    out = io.StringIO()
    cols = ["severity", "kind", "logger", "field_id", "start_date", "end_date", "title", "detail", "action"]
    w = csv.DictWriter(out, cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(qc["issues"])
    return out.getvalue()


def report_markdown(qc, label, week=None):
    """The weekly QC report as Markdown: one week (index into qc["weeks"]; default the last) or the season."""
    weeks = qc["weeks"]
    W = weeks[-1] if week is None else weeks[week]
    a, b = W["start"], W["end"]
    open_ = [i for i in qc["issues"] if i["start"] <= b and i["end"] >= a and i["severity"] != "info"]
    lines = [f"# Data quality report, {W['week']} ({W['start_date']} to {W['end_date']})", "", f"Campaign: {label}", "",
             "## This week", "",
             f"- Logger uptime: {100 * W['logger_uptime']:.0f} % of logger-days",
             "- Satellite calls (field-passes): " + ", ".join(f"{SENSOR_NAME.get(k, k)} {v}" for k, v in W["satellite_calls"].items()),
             f"- Logged irrigations: {W['logged_irrigations']} by {W['farmers_logging']} farmers"]
    if W["metered_m3"]:
        lines.append(f"- Metered water {W['metered_m3']:.0f} m³; logged {W['logged_m3']:.0f} m³ "
                     f"({100 * W['logged_m3'] / W['metered_m3']:.0f} %)")
    lines += ["", f"## Open issues ({len(open_)})", ""]
    if open_:
        lines += ["| Severity | Where | Issue | From | To | Action |", "|---|---|---|---|---|---|"]
        for i in open_:
            where = i["logger"] and f"logger {i['logger']}" or i["field_id"] and f"field {i['field_id']}" or "network"
            lines.append(f"| {i['severity']} | {where} | {i['title']}: {i['detail']} | {i['start_date']} | {i['end_date']} | {i['action']} |")
    else:
        lines.append("None.")
    lines += ["", "## Season so far", "", "| Week | Logger uptime | S2 | S1 | NISAR | Logged irrigations | New issues |",
              "|---|---|---|---|---|---|---|"]
    for w in weeks:
        if w["start"] > b:
            break
        sc = w["satellite_calls"]
        lines.append(f"| {w['week']} | {100 * w['logger_uptime']:.0f} % | {sc.get('s2', 0)} | {sc.get('s1', 0)} | {sc.get('nisar', 0)} "
                     f"| {w['logged_irrigations']} | {len(w['new_issues'])} |")
    info = [i for i in qc["issues"] if i["severity"] == "info"]
    if info:
        lines += ["", "## Notes", ""] + [f"- {i['title']}: {i['detail']}" for i in info]
    return "\n".join(lines) + "\n"
