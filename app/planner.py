"""Logger placement planner: which plots get the campaign's water-level loggers.

In December the campaign must choose 40-60 plots for loggers. A plan is judged on what the loggers have to do:

  calibration   at least 20 working loggers make the S&P and forecast intervals honest (Experiment 19); with
                a failure rate per logger, the chance that enough survive the season, and the plan size that
                makes it 90 % or 95 % likely
  coverage      the loggers' share in each stratum against the plots' share: tubewell, position along the
                channel (head, middle, tail), plot size, and every attribute in the site file (soil,
                smartphone, ...); a stratum with plots but no logger is called out
  pairs         adjacent logger plots (sharing a bund), for the water moving between plots
  upkeep        each plot's distance to the nearest logger, the loggers' distance from a road, and the walk
                to visit them all (the weekly read-out)

`suggest` builds a plan greedily and then improves it by swaps, keeping the plots the user locked and never
using a plot whose farmer said no (attribute consent = "no") or that the user excluded. `evaluate` compares a
plan with random plans of the same size, so "well balanced" has a reference.
"""

import csv
import io
import json
import math

import numpy as np

# Experiment 19 (random-mismatch world): mean coverage of the 90 % intervals and the 10-90 % range over draws,
# by the number of logger fields used for calibration
EXP19 = [  # n, S&P mean, S&P p10, S&P p90, forecast mean, forecast p10, forecast p90
    (3, 0.79, 0.56, 0.98, 0.83, 0.66, 0.97), (5, 0.81, 0.61, 0.98, 0.89, 0.77, 0.99),
    (10, 0.91, 0.77, 1.00, 0.95, 0.89, 0.99), (20, 0.95, 0.87, 0.99, 0.91, 0.87, 0.98),
    (40, 0.94, 0.88, 0.98, 0.90, 0.86, 0.94), (80, 0.92, 0.87, 0.97, 0.90, 0.88, 0.93),
    (160, 0.91, 0.88, 0.94, 0.90, 0.89, 0.92)]
NEED = 20                 # working loggers for honest intervals (Experiment 19)
ADJ_TOL, ADJ_MIN = 3.5, 10.0   # plots whose boundaries run within 3.5 m of each other for at least 10 m share a bund
EDGE_KEEP = 6.0           # a pipe at least this far inside the bunds (or as far as the plot allows)
WALK_KMH, READ_MIN = 4.0, 10.0
MAX_LEVELS = 8            # an attribute with more distinct values is not a stratum (free text, ids)


# ── geometry ──────────────────────────────────────────────────────────────────
def _seg_dist(p, a, b):
    """Distance from points p (k, 2) to the segments a-b (m, 2) each: (k, m)."""
    ab = b - a
    t = np.clip(((p[:, None, :] - a[None]) * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9)[None], 0, 1)
    q = a[None] + t[..., None] * ab[None]
    return np.sqrt(((p[:, None, :] - q) ** 2).sum(-1))


def _edges(poly):
    P = np.asarray(poly, float)
    return P, np.roll(P, -1, axis=0)


def _sample_boundary(poly, step=1.0):
    a, b = _edges(poly)
    pts = []
    for p0, p1 in zip(a, b):
        k = max(1, int(np.linalg.norm(p1 - p0) / step))
        pts.append(p0 + (p1 - p0) * (np.arange(k)[:, None] + 0.5) / k)
    return np.vstack(pts), step


def _inside(x, z, poly):
    inside = False
    for (x0, z0), (x1, z1) in zip(poly, poly[1:] + poly[:1]):
        if (z0 > z) != (z1 > z) and x < x0 + (z - z0) * (x1 - x0) / (z1 - z0):
            inside = not inside
    return inside


def logger_point(poly, c):
    """Where the pipe goes: inside the plot, at least EDGE_KEEP m from every bund if possible, else as far in as
    the plot allows; among those, the point nearest the plot's centroid."""
    P = np.asarray(poly, float)
    x0, z0 = P.min(0)
    x1, z1 = P.max(0)
    xs, zs = np.meshgrid(np.linspace(x0, x1, 25), np.linspace(z0, z1, 25))
    cand = np.c_[xs.ravel(), zs.ravel()]
    cand = cand[[_inside(x, z, poly) for x, z in cand]]
    if not len(cand):
        return [round(c[0], 2), round(c[1], 2)]
    a, b = _edges(poly)
    d_edge = _seg_dist(cand, a, b).min(1)
    keep = np.minimum(d_edge, EDGE_KEEP)
    ok = keep >= keep.max() - 0.25
    d_c = np.hypot(cand[:, 0] - c[0], cand[:, 1] - c[1])
    k = np.flatnonzero(ok)[np.argmin(d_c[ok])]
    return [round(float(cand[k, 0]), 2), round(float(cand[k, 1]), 2)]


def _polyline_dist(p, lines):
    """Distance from points p (k, 2) to the nearest of several polylines, and the nearest point."""
    best = np.full(len(p), np.inf)
    near = np.zeros((len(p), 2))
    for pts in lines:
        L = np.asarray(pts, float)
        if len(L) < 2:
            continue
        a, b = L[:-1], L[1:]
        ab = b - a
        t = np.clip(((p[:, None, :] - a[None]) * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9)[None], 0, 1)
        q = a[None] + t[..., None] * ab[None]
        d = np.sqrt(((p[:, None, :] - q) ** 2).sum(-1))
        j = d.argmin(1)
        dj = d[np.arange(len(p)), j]
        better = dj < best
        best[better] = dj[better]
        near[better] = q[np.arange(len(p)), j][better]
    return best, near


def _along(p, pts):
    """Distance along a polyline (from its first point) of each point's nearest position on it."""
    L = np.asarray(pts, float)
    a, b = L[:-1], L[1:]
    ab = b - a
    seg = np.linalg.norm(ab, axis=1)
    cum = np.r_[0, np.cumsum(seg)][:-1]
    t = np.clip(((p[:, None, :] - a[None]) * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9)[None], 0, 1)
    q = a[None] + t[..., None] * ab[None]
    d = np.sqrt(((p[:, None, :] - q) ** 2).sum(-1))
    j = d.argmin(1)
    return cum[j] + t[np.arange(len(p)), j] * seg[j], d[np.arange(len(p)), j]


def _terciles(v, names):
    v = np.asarray(v, float)
    if len(v) < 3 or np.ptp(v) == 0:
        return [names[1]] * len(v)
    q1, q2 = np.quantile(v, [1 / 3, 2 / 3])
    return [names[0] if x <= q1 else names[1] if x <= q2 else names[2] for x in v]


# ── the site as the planner sees it ───────────────────────────────────────────
def site_info(lay):
    """Per-plot strata, adjacency, eligibility, pipe position and road distance. Cached on the layout dict."""
    if "_planner" in lay:
        return lay["_planner"]
    F = lay["fields"]
    n = len(F)
    cents = np.array([f["c"] for f in F], float)
    tws = lay["tubewells"]

    # position along the channel from the tubewell: head, middle, tail (terciles within each tubewell)
    pos = np.full(n, np.nan)
    for t in range(len(tws)):
        idx = [i for i in range(n) if F[i]["tw"] == t]
        if not idx:
            continue
        chans = [c_["pts"] for c_ in lay.get("channels", []) if c_["tw"] == t and len(c_["pts"]) >= 2]
        P = cents[idx]
        if chans:
            best = np.full(len(idx), np.inf)
            for pts in chans:
                start = np.asarray(pts[0], float)
                if np.hypot(*(np.asarray(pts[-1]) - [tws[t]["x"], tws[t]["z"]])) < np.hypot(*(start - [tws[t]["x"], tws[t]["z"]])):
                    pts = pts[::-1]
                al, d = _along(P, pts)
                use = d < best
                best[use] = d[use]
                pos[np.array(idx)[use]] = al[use]
        else:
            pos[idx] = np.hypot(P[:, 0] - tws[t]["x"], P[:, 1] - tws[t]["z"])
    channel = [""] * n
    for t in range(len(tws)):
        idx = [i for i in range(n) if F[i]["tw"] == t]
        for i, lab in zip(idx, _terciles(pos[idx], ["head", "middle", "tail"])):
            channel[i] = lab
    size = _terciles([f["area_m2"] for f in F], ["small", "medium", "large"])

    strata = {"tubewell": [tws[f["tw"]]["name"] if f["tw"] < len(tws) else str(f["tw"]) for f in F],
              "channel position": channel, "plot size": size}
    strata = {k: v for k, v in strata.items() if len(set(v)) >= 2}   # e.g. equal plots on the generated grid
    keys = sorted({k for f in F for k in f.get("attrs", {}) if k != "consent"})
    for k in keys:
        vals = [str(f.get("attrs", {}).get(k, "unknown")) for f in F]
        if 2 <= len(set(vals)) <= MAX_LEVELS:
            strata[k] = vals
    consent = [str(f.get("attrs", {}).get("consent", "")).lower() for f in F]

    # adjacency: plots that share a bund
    boxes = np.array([[*np.min(f["poly"], 0), *np.max(f["poly"], 0)] for f in F])
    samples = [_sample_boundary(f["poly"]) for f in F]
    adj = {}
    for i in range(n):
        for j in range(i + 1, n):
            if (boxes[i, 0] > boxes[j, 2] + ADJ_TOL or boxes[j, 0] > boxes[i, 2] + ADJ_TOL
                    or boxes[i, 1] > boxes[j, 3] + ADJ_TOL or boxes[j, 1] > boxes[i, 3] + ADJ_TOL):
                continue
            pts, step = samples[i]
            a, b = _edges(F[j]["poly"])
            shared = float((_seg_dist(pts, a, b).min(1) <= ADJ_TOL).sum() * step)
            if shared >= ADJ_MIN:
                adj.setdefault(i, {})[j] = shared
                adj.setdefault(j, {})[i] = shared

    pipe = np.array([logger_point(f["poly"], f["c"]) for f in F])
    roads = [r["pts"] for r in lay.get("roads", [])]
    if roads:
        road_d, road_pt = _polyline_dist(pipe, roads)
    else:
        road_d = np.full(n, np.nan)
        road_pt = pipe.copy()
    info = {"n": n, "strata": strata, "consent": consent, "eligible": np.array([c_ != "no" for c_ in consent]),
            "adj": adj, "pipe": pipe, "road_d": road_d, "road_pt": road_pt, "cents": cents,
            "levels": {k: sorted(set(v)) for k, v in strata.items()}}
    info["codes"] = {k: np.array([info["levels"][k].index(x) for x in v]) for k, v in strata.items()}
    lay["_planner"] = info
    return info


# ── scoring ───────────────────────────────────────────────────────────────────
def _pairs(sel, adj):
    """Disjoint adjacent pairs within the selection (greedy, longest shared bund first)."""
    s = set(sel)
    cand = sorted(((adj[i][j], i, j) for i in s for j in adj.get(i, {}) if j in s and i < j), reverse=True)
    used, out = set(), []
    for L, i, j in cand:
        if i not in used and j not in used:
            used |= {i, j}
            out.append((i, j, L))
    return out


def _balance(sel, info):
    """Per stratum dimension, 1 - total variation distance between the loggers' and the plots' level shares."""
    out = {}
    for k, codes in info["codes"].items():
        m = len(info["levels"][k])
        pf = np.bincount(codes, minlength=m) / len(codes)
        pl = np.bincount(codes[sel], minlength=m) / max(1, len(sel))
        out[k] = 1 - 0.5 * float(np.abs(pf - pl).sum())
    return out


def _nn(sel, info):
    """Each plot's distance (m) to the nearest logger pipe."""
    if not len(sel):
        return np.full(info["n"], np.inf)
    d = np.hypot(info["cents"][:, None, 0] - info["pipe"][None, sel, 0], info["cents"][:, None, 1] - info["pipe"][None, sel, 1])
    return d.min(1)


def _objective(sel, info, pairs_target, ref):
    sel = np.asarray(sel, int)
    bal = _balance(sel, info)
    b = float(np.mean(list(bal.values())))
    pr = min(1.0, len(_pairs(sel, info["adj"])) / pairs_target) if pairs_target else 0.0
    spread = 1 - min(1.0, float(np.median(_nn(sel, info))) / ref["nn"])
    rd = info["road_d"][sel]
    access = 1 - min(1.0, float(np.nanmean(rd)) / ref["road"]) if np.isfinite(rd).any() else 0.0
    return 3.0 * b + 1.0 * pr + 0.5 * spread + 0.25 * access


def _refs(info):
    D = np.hypot(info["cents"][:, None, 0] - info["cents"][None, :, 0], info["cents"][:, None, 1] - info["cents"][None, :, 1])
    rd = info["road_d"]
    return {"nn": max(1.0, float(np.median(D))), "road": max(1.0, float(np.nanmax(rd))) if np.isfinite(rd).any() else 1.0}


def suggest(lay, n, locked=(), excluded=(), pairs_target=10, seed=0, swaps=2):
    """A plan of n plots: the locked ones, plus a greedy then swap-improved choice among the eligible plots."""
    info = site_info(lay)
    N = info["n"]
    n = int(max(1, min(n, N)))
    locked = [int(i) for i in locked if 0 <= int(i) < N]
    bad = set(int(i) for i in excluded) | set(np.flatnonzero(~info["eligible"]).tolist())
    pool = [i for i in range(N) if i not in bad and i not in locked]
    rng = np.random.default_rng(seed)
    rng.shuffle(pool)
    ref = _refs(info)
    sel = list(locked)
    while len(sel) < n and pool:
        scores = [_objective(sel + [c], info, pairs_target, ref) for c in pool]
        c = pool.pop(int(np.argmax(scores)))
        sel.append(c)
    best = _objective(sel, info, pairs_target, ref)
    for _ in range(swaps):
        improved = False
        for k in range(len(locked), len(sel)):
            for c in list(pool):
                trial = sel[:k] + [c] + sel[k + 1:]
                v = _objective(trial, info, pairs_target, ref)
                if v > best + 1e-9:
                    pool.remove(c)
                    pool.append(sel[k])
                    sel, best, improved = trial, v, True
        if not improved:
            break
    return sorted(sel)


def _exp19(k):
    """Experiment 19's expected interval coverage for k working loggers (interpolated in log k)."""
    if k < EXP19[0][0]:
        return None
    rows = np.array(EXP19)
    x = np.log(np.clip(k, rows[0, 0], rows[-1, 0]))
    return {name: round(float(np.interp(x, np.log(rows[:, 0]), rows[:, j])), 2)
            for j, name in enumerate(["sp", "sp_p10", "sp_p90", "fc", "fc_p10", "fc_p90"], start=1)}


def _p_at_least(n, p_fail, need):
    q = 1 - p_fail
    return float(sum(math.comb(n, k) * q ** k * p_fail ** (n - k) for k in range(need, n + 1))) if n >= need else 0.0


def _route(points, start):
    """A visiting walk from a road point through every pipe and back: nearest neighbour, then 2-opt. Metres."""
    if not len(points):
        return 0.0, []
    P = np.vstack([start, points])
    D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
    tour, left = [0], set(range(1, len(P)))
    while left:
        j = min(left, key=lambda k: D[tour[-1], k])
        tour.append(j)
        left.remove(j)
    tour.append(0)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(tour) - 2):
            for j in range(i + 1, len(tour) - 1):
                if D[tour[i - 1], tour[j]] + D[tour[i], tour[j + 1]] < D[tour[i - 1], tour[i]] + D[tour[j], tour[j + 1]] - 1e-6:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
    length = float(sum(D[a, b] for a, b in zip(tour, tour[1:])))
    return length, [int(k - 1) for k in tour[1:-1]]


def evaluate(lay, sel, p_fail=0.1, pairs_target=10, n_random=200, seed=0):
    info = site_info(lay)
    sel = sorted(set(int(i) for i in sel if 0 <= int(i) < info["n"]))
    S = np.asarray(sel, int)
    n = len(sel)
    exp_work = n * (1 - p_fail)
    need_for = {}
    for target in (0.9, 0.95):
        k = NEED
        while _p_at_least(k, p_fail, NEED) < target and k < 400:
            k += 1
        need_for[str(target)] = k

    strata = []
    for k, codes in info["codes"].items():
        lv = info["levels"][k]
        nf = np.bincount(codes, minlength=len(lv))
        nl = np.bincount(codes[S], minlength=len(lv)) if n else np.zeros(len(lv), int)
        strata.append({"name": k, "levels": [{"level": lv[i], "plots": int(nf[i]), "loggers": int(nl[i]),
                                              "expected": round(n * nf[i] / info["n"], 1)} for i in range(len(lv))]})
    bal = _balance(S, info) if n else {k: 0.0 for k in info["codes"]}
    missing = [f"{s['name']}: {lv['level']}" for s in strata for lv in s["levels"]
               if lv["loggers"] == 0 and lv["plots"] >= max(3, 0.05 * info["n"]) and n >= 5]

    pairs = _pairs(S, info["adj"])
    F = lay["fields"]
    nn = _nn(S, info)
    rd = info["road_d"][S] if n else np.array([])
    if n and np.isfinite(info["road_d"]).any():
        start = info["road_pt"][S[np.nanargmin(info["road_d"][S])]]
    elif n:
        start = info["pipe"][S[0]]
    route_m, order = _route(info["pipe"][S], start) if n else (0.0, [])

    # how this plan's balance compares with random plans of the same size (from the eligible plots)
    rng = np.random.default_rng(seed)
    elig = np.flatnonzero(info["eligible"])
    rand_bal, rand_pairs = [], []
    if n and n <= len(elig):
        for _ in range(n_random):
            R = rng.choice(elig, n, replace=False)
            rand_bal.append(np.mean(list(_balance(R, info).values())))
            rand_pairs.append(len(_pairs(R, info["adj"])))
    mb = float(np.mean(list(bal.values()))) if n else 0.0
    return {
        "n": n, "fields": sel, "p_fail": p_fail, "need": NEED,
        "calibration": {"expected_working": round(exp_work, 1), "p_enough": round(_p_at_least(n, p_fail, NEED), 3),
                        "plan_size_for": need_for, "exp19": _exp19(exp_work)},
        "strata": strata, "balance": {k: round(v, 3) for k, v in bal.items()}, "balance_mean": round(mb, 3),
        "balance_vs_random": round(float(np.mean(np.array(rand_bal) < mb)), 2) if rand_bal else None,
        "random_balance_median": round(float(np.median(rand_bal)), 3) if rand_bal else None,
        "missing": missing,
        "pairs": [{"a": int(i), "b": int(j), "a_id": F[i]["id"], "b_id": F[j]["id"], "shared_m": round(L, 0)} for i, j, L in pairs],
        "pairs_target": pairs_target, "random_pairs_median": float(np.median(rand_pairs)) if rand_pairs else None,
        "available_pairs": sum(len(v) for v in info["adj"].values()) // 2,
        "spread": {"median_m": round(float(np.median(nn)), 0) if n else None, "max_m": round(float(nn.max()), 0) if n else None,
                   "far_plots": int((nn > 150).sum()) if n else info["n"]},
        "access": {"median_road_m": round(float(np.nanmedian(rd)), 0) if n and np.isfinite(rd).any() else None,
                   "max_road_m": round(float(np.nanmax(rd)), 0) if n and np.isfinite(rd).any() else None},
        "route": {"length_m": round(route_m, 0), "hours": round(route_m / 1000 / WALK_KMH + n * READ_MIN / 60, 1),
                  "order": [sel[k] for k in order]},
        "ineligible": [F[i]["id"] for i in sel if not info["eligible"][i]],
        "pipes": {str(i): info["pipe"][i].tolist() for i in sel},
    }


def site_summary(lay):
    """What the front end needs to draw the planner: strata per plot, eligibility, pipe positions, adjacency."""
    info = site_info(lay)
    return {"strata": info["strata"], "levels": info["levels"], "consent": info["consent"],
            "eligible": info["eligible"].tolist(), "pipes": info["pipe"].round(2).tolist(),
            "adjacent": {str(i): sorted(v) for i, v in info["adj"].items()},
            "current": sorted({lg["field"] for lg in lay.get("loggers", [])})}


# ── layouts for the planner (its own copies: site_info caches arrays on them) ───
_layouts = {}


def get_layout(site, n_grid=120):
    try:
        from . import sites
    except ImportError:  # imported as a top-level module (tests)
        import sites
    key = (site, int(n_grid) if site == "grid" else 0)
    if key not in _layouts:
        if len(_layouts) > 8:
            _layouts.clear()
        _layouts[key] = sites.layout(site, int(n_grid))
    return _layouts[key]


# ── export ────────────────────────────────────────────────────────────────────
def _lonlat(lay, x, z):
    lon0, lat0 = lay.get("origin_lonlat") or (None, None)
    if lon0 is None:
        return None, None
    kx = math.radians(1) * 6_371_000.0 * math.cos(math.radians(lat0))
    kz = math.radians(1) * 6_371_000.0
    return round(lon0 + x / kx, 7), round(lat0 - z / kz, 7)


def export_csv(lay, sel):
    ev = evaluate(lay, sel, n_random=0)
    info = site_info(lay)
    partner = {}
    for p in ev["pairs"]:
        partner[p["a"]], partner[p["b"]] = p["b_id"], p["a_id"]
    visit = {f: k + 1 for k, f in enumerate(ev["route"]["order"])}
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["logger_id", "field_id", "visit_order", "lon", "lat", "x_m", "z_m", "pair_with", "road_m", *info["strata"].keys()])
    for k, f in enumerate(sorted(ev["fields"], key=lambda f: visit.get(f, 0))):
        x, z = info["pipe"][f]
        lon, lat = _lonlat(lay, x, z)
        w.writerow([f"P{k + 1:02d}", lay["fields"][f]["id"], visit.get(f, ""), lon, lat, round(x, 1), round(z, 1),
                    partner.get(f, ""), None if not np.isfinite(info["road_d"][f]) else round(float(info["road_d"][f]), 0),
                    *[v[f] for v in info["strata"].values()]])
    return out.getvalue()


def export_geojson(lay, sel, site_path=None):
    """The site file with its loggers replaced by the plan's (planned pipe positions), as a new site."""
    info = site_info(lay)
    if site_path is None:
        raise ValueError("the generated grid has no site file to export; pick a site")
    gj = json.loads(site_path.read_text())
    feats = [f for f in gj["features"] if f.get("properties", {}).get("kind") != "logger"]
    for k, f in enumerate(sorted(set(int(i) for i in sel))):
        lon, lat = _lonlat(lay, *info["pipe"][f])
        feats.append({"type": "Feature", "properties": {"kind": "logger", "id": f"P{k + 1:02d}", "field": lay["fields"][f]["id"], "planned": True},
                      "geometry": {"type": "Point", "coordinates": [lon, lat]}})
    props = dict(gj.get("properties", {}))
    props["name"] = f"{props.get('name', 'site')}_plan"
    props["label"] = f"{props.get('label', 'site')}: logger plan"
    return json.dumps({**gj, "properties": props, "features": feats}, indent=1)
