"""Paddy Twin engine: one simulated season of a village, and what KGML-Paddy believes about it each day.

Everything here is the research code (awdrt.paddy_sim simulator v2, awdrt.kgml_paddy v1b) run live:
  - the simulator gives each field's true water level, irrigations, satellite passes and farmer log;
  - the model is run causally: its belief at the end of day d uses observations up to day d only
    (one cut per day, batched), so the X-ray view shows what the model would have said in real time;
  - each day, a "dry by tomorrow?" alert from the next-dry forecast (Experiment 11);
  - each Sentinel-2 pass is checked against the model's prior (N3, Experiment 14);
  - a farmer's "is there water now?" answer updates one field's belief (Experiment 13).
All numbers are simulation, not field data.
"""

import pathlib
import sys
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import replace

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from awdrt import counterfactual as cf  # noqa: E402
from awdrt import kgml_paddy as kp  # noqa: E402
from awdrt import methane as mh  # noqa: E402
from awdrt import paddy_sim as ps  # noqa: E402

try:
    from . import campaign, qc, sites  # noqa: E402
except ImportError:  # imported as a top-level module (scripts, tests)
    import campaign  # noqa: E402
    import qc  # noqa: E402
    import sites  # noqa: E402

MODEL_PATH = ROOT / "results" / "models" / "kgml_paddy_v1b_pretrained.pt"
# the pre-trained models the app can run: v1b (default) and v1c, pre-trained on a wider S&P prior (Experiment 22)
MODELS = {"v1b": MODEL_PATH, "v1c": ROOT / "results" / "models" / "kgml_paddy_v1c_pretrained.pt",
          "v1d": ROOT / "results" / "models" / "kgml_paddy_v1d_pretrained.pt"}
MODELS = {k: v for k, v in MODELS.items() if v.is_file()}
# The twin grows its rice as Boro rice grows: in thermal time, from the run's own temperature (Experiment 27).
# v1d is pre-trained on that crop; v1b and v1c on the straight-line crop of Experiments 1-26.
ps.set_phenology("boro")
DEFAULT_MODEL = "v1d" if "v1d" in MODELS else "v1b"
_compute_lock = threading.RLock()   # one computation at a time: the crop's temperature series is per run
PROGRESS = OrderedDict()            # client token -> where a run is (GET /api/progress/<token>)


class _Progress:
    """Reports a run's stage and overall fraction under a token the page polls; a no-op without a token."""

    def __init__(self, token, stages):
        self.token, self.names = token, [n for n, _ in stages]
        w = np.array([float(x) for _, x in stages])
        self.start, self.w, self.i = np.r_[0, np.cumsum(w)[:-1]] / w.sum(), w / w.sum(), 0
        self.t0 = time.time()
        self.stage(0)

    def _put(self, frac):
        if not self.token:
            return
        PROGRESS[self.token] = {"stages": self.names, "stage": self.i, "frac": round(float(frac), 4),
                                "seconds": round(time.time() - self.t0, 1), "done": False}
        while len(PROGRESS) > 30:
            PROGRESS.popitem(last=False)

    def stage(self, i):
        self.i = i
        self._put(self.start[i])

    def sub(self, x):
        self._put(self.start[self.i] + self.w[self.i] * min(max(x, 0.0), 1.0))

    def done(self):
        if self.token:
            self._put(1.0)
            PROGRESS[self.token]["done"] = True


SEASON_STAGES = [("Laying out the village and simulating the season", 10), ("The model's belief, day by day", 45),
                 ("Forecasting when each field runs dry", 25), ("True forecasts, satellite checks and the crop", 12),
                 ("Packing the results", 8)]
REPLAY_STAGES = [("Reading the campaign files", 10), ("The model's belief, day by day", 45),
                 ("Forecasting when each field runs dry", 25), ("Logger truth, satellite checks and data QC", 12),
                 ("Packing the results", 8)]


def _locked(fn):
    def run(*a, **k):
        with _compute_lock:
            return fn(*a, **k)
    run.__name__, run.__doc__ = fn.__name__, fn.__doc__
    return run
K = 24               # posterior samples per field-day for the forecast
K_ASK = 400          # samples for a farmer-answer update
MAX_DAYS = 15.0      # forecast horizon (weather data ends 16 days after the season)
FC_DT = 0.1
ALERT_DAYS = 1.0     # "irrigate by tomorrow": predicted dry within a day
NISAR_FIRST, NISAR_REVISIT = 5, 12   # as simulator v2 (paddy_sim.simulate, cfg.nisar)
COLS = 12            # generated grid: fields per map row (layout in sites.grid_layout)

LIMITS = {"n_fields": (24, 240), "et0_scale": (0.6, 1.6), "rain_scale": (0.0, 3.0), "leak": (1.5, 15.0),
          "p_awd": (0.0, 1.0), "p_topup": (0.0, 1.0), "pump_bias": (0.7, 1.5), "cloud": (0.0, 0.95),
          "share_logging": (0.0, 1.0), "seed": (0, 10_000), "nisar_acc": (0.55, 0.98), "nisar_assumed": (0.55, 0.98)}
DEFAULTS = {"n_fields": 120, "et0_scale": 1.0, "rain_scale": 1.0, "leak": 5.0, "p_awd": 0.2, "p_topup": 0.15,
            "pump_bias": 1.0, "cloud": 0.0, "share_logging": 0.7, "world": "prior", "seed": 1, "site": "grid",
            "nisar_acc": 0.8, "nisar_assumed": 0.8, "nisar_update": True, "model": DEFAULT_MODEL}

_models = {}
_model_lock = threading.Lock()
_runs = OrderedDict()  # run id -> run (last few kept for farmer questions)


def _bacc(truth, pred):
    t, q = np.asarray(truth, bool), np.asarray(pred, bool)
    if t.all() or (~t).all():
        return None
    return float(0.5 * ((t & q).sum() / t.sum() + (~t & ~q).sum() / (~t).sum()))


def model(name="v1b"):
    with _model_lock:
        if name not in _models:
            m = kp.PaddyNet()
            m.load_state_dict(torch.load(MODELS[name]))
            m.eval()
            _models[name] = m
    return _models[name]


def clean_params(p):
    out = dict(DEFAULTS)
    out.update({k: v for k, v in (p or {}).items() if k in DEFAULTS})
    for k, (lo, hi) in LIMITS.items():
        out[k] = min(max(float(out[k]), lo), hi)
    out["n_fields"] = int(round(out["n_fields"] / COLS)) * COLS
    out["seed"] = int(out["seed"])
    out["nisar_update"] = bool(out["nisar_update"]) and out["nisar_update"] not in ("0", "false", "False")
    if out["p_awd"] + out["p_topup"] > 1:
        out["p_topup"] = 1 - out["p_awd"]
    if out["world"] not in ps.WORLDS:
        out["world"] = "prior"
    if out["site"] not in {s["name"] for s in sites.list_sites()}:
        out["site"] = "grid"
    if out["model"] not in MODELS:
        out["model"] = DEFAULT_MODEL
    return out


def _crop(tp_abs, g, T, s0):
    """The crop on each field and day (end of day): height (cm), tillers per hill, stage, ripening (0-1)."""
    days = np.arange(T)[None, :] + s0 + 0.5
    c = ps.crop_state(days, np.asarray(tp_abs, float)[:, None], np.asarray(g, float)[:, None])
    ripen = np.clip((c["tt"] - ps.BORO["tt_head"]) / (ps.BORO["tt_mature"] - ps.BORO["tt_head"]), 0, 1)
    return c, ripen


def _nisar_grid(acc, crop, season):
    """NISAR's accuracy on each field and day from the canopy's height (paddy_sim.nisar_accuracy)."""
    return np.where(season, ps.nisar_accuracy(acc, crop["height_cm"]), acc)


def _crop_payload(crop, ripen, season):
    z = lambda a, r=0: np.round(np.where(season, a, 0), r).tolist()  # noqa: E731
    return {"height_cm": z(crop["height_cm"]), "tillers": z(crop["tillers"]),
            "stage": np.where(season, crop["stage"], -1).astype(int).tolist(), "ripen": z(ripen, 2), "cover": z(crop["cover"], 2),
            "stages": list(ps.STAGES), "phenology": ps.PHENOLOGY}


def _sensing_tables(inp, nisar_acc):
    """What the field panel needs to say how well each satellite sees water under a canopy: Sentinel-2's
    recall by LAI (Experiment 1), Sentinel-1's accuracy by LAI (03b), NISAR's accuracy by plant height."""
    return {"s2_lai_lo": [float(x) for x in inp["lai_lo"]], "s2_recall": [round(float(x), 3) for x in inp["rec"]],
            "s1_lai_lo": [float(b[0]) for b in inp["s1_bins"]], "s1_acc": [round(float(b[1]), 3) for b in inp["s1_bins"]],
            "nisar_acc": float(nisar_acc), "nisar_full_cm": 70.0, "nisar_drop": 0.15, "nisar_drop_cm": 30.0}


def _inputs(p, base):
    return {**base, "et0": base["et0"] * p["et0_scale"], "rain": base["rain"] * p["rain_scale"],
            "s2_valid": base["s2_valid"] * (1 - p["cloud"])}


def _forecast(theta, h, t0, tp, inp, progress=None):
    return kp.forecast_dry(theta, h, t0, tp, inp, max_days=MAX_DAYS, dt=FC_DT, progress=progress)


def _belief(X, static, season, tp_abs, inp, seed, nisar_said, nisar_update, nisar_assumed, model_name="v1b", prog=None):
    """The model's causal day-by-day belief and next-dry forecast, with the NISAR answer update.
    Shared by the simulation and replay modes. Arrays are flat over (day, field): index d * n + f."""
    n, T = X.shape[:2]
    # causal belief: for day d, observations through day d (cut = d + 1), state at the end of day d
    cut = np.repeat(np.arange(1, T + 1), n)
    fld = np.tile(np.arange(n), T)
    th, st = kp.predict_v1(model(model_name), X[fld], static[fld], cut, n_samples=K, seed=seed, progress=prog.sub if prog else None)
    day = cut - 1
    pick = lambda a: a[np.arange(len(cut)), day]  # noqa: E731
    pw, pmu, psd, nmu, nsd = (pick(st[k]) for k in ("p_wet", "pos_mu", "pos_sd", "neg_mu", "neg_sd"))
    srng = np.random.default_rng(seed + 2)
    wet = srng.random((K, len(cut))) < pw
    hs = np.where(wet, np.maximum(pmu + psd * srng.standard_normal(wet.shape), 0.1),
                  np.clip(nmu + nsd * srng.standard_normal(wet.shape), ps.PIPE_MIN, 0.0))
    in_season = season[fld, day]
    # the NISAR answer update (Experiment 24). The belief and forecast of day d are the advice for the morning
    # of day d + 1; a NISAR pass at 06:00 that morning reweights them by Bayes' rule, as a yes/no answer with
    # the assumed accuracy (Experiment 24's timing exactly: data before day d + 1, pass on day d + 1).
    W = np.ones_like(hs)
    nisar_next = np.full((n, T), -1, np.int8)
    nisar_next[:, :-1] = nisar_said[:, 1:]
    said_flat = nisar_next.T.reshape(-1)                    # index d * n + f, as cut / fld
    upd = (said_flat >= 0) & in_season & nisar_update
    iu = np.flatnonzero(upd)
    if len(iu):
        acc = np.asarray(nisar_assumed, float)
        if acc.ndim == 2:    # by field and pass day (the canopy's height): the pass of day d + 1 updates day d
            nxt = np.full((n, T), 0.8)
            nxt[:, :-1] = acc[:, 1:]
            acc = nxt.T.reshape(-1)[iu]
        W[:, iu] = kp.answer_likelihood(hs[:, iu], said_flat[iu], "yesno", p_correct=acc)
    pw_model = pw
    pw = np.where(upd, (W * (hs > 0)).sum(0) / W.sum(0), pw)
    q05, q50, q95 = np.quantile(hs, [0.05, 0.5, 0.95], 0)
    expect = np.maximum(np.where(pw > 0.5, pmu, nmu), ps.PIPE_MIN)
    if len(iu):
        q05[iu], q50[iu], q95[iu] = kp.weighted_quantiles(hs[:, iu], W[:, iu], [0.05, 0.5, 0.95])
        expect[iu] = q50[iu]

    # next-dry forecast and the "irrigate by tomorrow" alert, for fields the model thinks are ponded
    t0 = inp["s0"] + day + 1.0
    tp = tp_abs[fld]
    D_pred = np.full(len(cut), np.nan)
    D_before = np.full(len(cut), np.nan)   # without the NISAR update (for the statistics)
    sel = np.flatnonzero(in_season & ((pw > 0.5) | (pw_model > 0.5)))
    if len(sel):
        if prog:
            prog.stage(2)
        Ds = _forecast(th[:, sel], hs[:, sel], t0[sel], tp[sel][None, :], inp, progress=prog.sub if prog else None)
        D_before[sel] = np.median(Ds, 0)
        D_pred[sel] = D_before[sel]
        su = np.flatnonzero(upd[sel])
        if len(su):
            D_pred[sel[su]] = kp.weighted_quantiles(Ds[:, su], W[:, sel[su]], [0.5])[0]
    D_pred[pw <= 0.5] = np.nan
    D_before[pw_model <= 0.5] = np.nan
    return {"cut": cut, "fld": fld, "day": day, "th": th, "pw": pw, "pw_model": pw_model, "in_season": in_season,
            "upd": upd, "iu": iu, "q05": q05, "q50": q50, "q95": q95, "expect": expect, "D_pred": D_pred,
            "D_before": D_before, "t0": t0, "tp": tp}


@_locked
def run_season(params, base_inp, token=None):
    """Simulate the season and the model's day-by-day belief. Returns a JSON-ready dict. With a token, the run's
    progress is readable in PROGRESS while it runs."""
    t_start = time.time()
    prog = _Progress(token, SEASON_STAGES)
    ps.set_temperature(base_inp["tmean"])
    p = clean_params(params)
    inp = _inputs(p, base_inp)
    w0 = ps.WORLDS[p["world"]]
    cfg = replace(w0, a_median=w0.a_median * p["leak"] / 5.0, p_awd=p["p_awd"], p_topup=p["p_topup"],
                  pump_bias_median=w0.pump_bias_median * p["pump_bias"], nisar_acc=p["nisar_acc"])
    lay = sites.layout(p["site"], p["n_fields"])
    n = len(lay["fields"])
    n_sim = n + n % 2  # the simulator works in neighbour pairs
    sim = ps.simulate(cfg, n_sim, p["seed"], inp)
    if n_sim > n:
        sim = {k: (v[:n] if isinstance(v, np.ndarray) and v.shape[:1] == (n_sim,) else v) for k, v in sim.items()}
    if p["site"] != "grid":
        p["n_fields"] = n
    rng = np.random.default_rng(p["seed"] + 1)
    logs = rng.random(n) < p["share_logging"]
    X = kp.drop_logs(sim["X"], ~logs)
    H = ps.end_of_day_depth(sim["h6"])
    T = X.shape[1]
    season = X[:, :, kp.CH["season"]] > 0
    # NISAR (06:00, L-band SAR, every 12 days): drawn from the truth with the simulated accuracy, on its own
    # random stream. v1b was trained without NISAR; a pass enters as an answer update (below).
    nisar_said = np.full((n, T), -1, np.int8)
    nrng = np.random.default_rng(p["seed"] + 3)
    nisar_days = np.arange(NISAR_FIRST, T, NISAR_REVISIT)
    crop, ripen = _crop(sim["tp_abs"], sim["g"], T, inp["s0"])
    crop_seen, _ = _crop(sim["tp_abs"], np.ones(n), T, inp["s0"])    # what the app knows: the nominal crop
    acc_true = _nisar_grid(cfg.nisar_acc, crop, season)              # the taller the rice, the less NISAR sees
    acc_assumed = _nisar_grid(p["nisar_assumed"], crop_seen, season)
    for d_ in nisar_days:
        pf = np.flatnonzero(season[:, d_])
        truth_p = sim["h6"][pf, d_ * ps.PER_DAY + 1] > 0
        a_ = acc_true[pf, d_]
        nisar_said[pf, d_] = np.where(truth_p, nrng.random(len(pf)) < a_, nrng.random(len(pf)) > a_)

    prog.stage(1)
    b = _belief(X, sim["static"], season, sim["tp_abs"], inp, p["seed"], nisar_said, p["nisar_update"], acc_assumed, p["model"], prog)
    prog.stage(3)
    cut, fld, day, th, pw, pw_model, in_season, upd, iu = (b[k] for k in ("cut", "fld", "day", "th", "pw", "pw_model", "in_season", "upd", "iu"))
    q05, q50, q95, expect, D_pred, D_before, t0, tp = (b[k] for k in ("q05", "q50", "q95", "expect", "D_pred", "D_before", "t0", "tp"))
    h_true = H[fld, day]
    D_true = np.full(len(cut), np.nan)
    tsel = np.flatnonzero(in_season & np.isfinite(h_true))
    D_true[tsel] = kp.forecast_dry(sim["theta"][fld[tsel]], h_true[tsel], t0[tsel], tp[tsel], inp,
                                   et_f=sim["et_f"][fld[tsel]], g=sim["g"][fld[tsel]], max_days=MAX_DAYS, dt=FC_DT,
                                   progress=lambda x: prog.sub(0.7 * x))

    # satellite passes, each checked against the model's prior (end of the previous day) - N3.
    # Sentinel-2 (10:30, optical) and Sentinel-1 (18:00, C-band SAR) are the simulator's observations that
    # the model reads. NISAR (06:00, L-band SAR, every 12 days) is drawn here from the truth with the
    # simulator's assumed accuracy, on its own random stream: v1b was trained without NISAR, so it is shown
    # and checked, but not fed to the model.
    prior = np.full((n, T), 0.5)
    prior[:, 1:] = pw.reshape(T, n).T[:, :-1]
    lai_obs = X[:, :, kp.CH["lai_obs"]] * 5
    passes = []

    def add(sensor, pf, pd_, said, truth_p, rec, spec):
        pr = prior[pf, pd_]
        fused = kp.fuse_pass(pr, said, rec, spec)
        flag = (said & (pr < 0.1)) | (~said & (pr > 0.9))
        passes.extend({"s": sensor, "f": int(a), "d": int(b), "wet": bool(c), "truth": bool(e), "prior": round(float(g), 3),
                       "fused": round(float(h), 3), "flag": bool(k)}
                      for a, b, c, e, g, h, k in zip(pf, pd_, said, truth_p, pr, fused, flag))

    for sensor, q in (("s2", 2), ("s1", 3)):
        wet, dry = X[:, :, kp.CH[f"{sensor}_wet"]] > 0, X[:, :, kp.CH[f"{sensor}_dry"]] > 0
        pf, pd_ = np.nonzero(wet | dry)
        if len(pf):
            add(sensor, pf, pd_, wet[pf, pd_], sim["h6"][pf, pd_ * ps.PER_DAY + q] > 0,
                *kp.pass_rates(inp, sensor, lai_obs[pf, pd_]))
    for d_ in nisar_days:
        pf = np.flatnonzero(nisar_said[:, d_] >= 0)
        truth_p = sim["h6"][pf, d_ * ps.PER_DAY + 1] > 0
        acc = acc_assumed[pf, d_]
        add("nisar", pf, np.full(len(pf), d_), nisar_said[pf, d_] > 0, truth_p, acc, acc)
    for q_ in passes:
        if q_["s"] == "nisar":
            q_["update"] = bool(q_["d"] >= 1 and upd[(q_["d"] - 1) * n + q_["f"]])   # updated the advice of the evening before
    passes.sort(key=lambda q_: (q_["d"], q_["f"]))
    # what the update did, on the mornings it was applied (model's belief before vs after, against the truth)
    nisar_stats = {"enabled": p["nisar_update"], "n": int(len(iu)), "assumed": p["nisar_assumed"], "true": p["nisar_acc"]}
    ev = iu[np.isfinite(D_true[iu])] if len(iu) else iu
    if len(ev):
        wet_true = H[fld[ev], day[ev]] > 0
        nisar_stats.update({
            "ponded_bacc_before": _bacc(wet_true, pw_model[ev] > 0.5), "ponded_bacc_after": _bacc(wet_true, pw[ev] > 0.5),
            "dry_err_before": float(np.median(np.abs(np.nan_to_num(D_before[ev], nan=0.0) - D_true[ev]))),
            "dry_err_after": float(np.median(np.abs(np.nan_to_num(D_pred[ev], nan=0.0) - D_true[ev])))})
    sensors = {"s2": {"name": "Sentinel-2", "time": "10:30", "kind": "optical", "dir": "descending",
                      "days": sorted({q_["d"] for q_ in passes if q_["s"] == "s2"}), "model_input": True},
               "s1": {"name": "Sentinel-1", "time": "18:00", "kind": "C-band SAR", "dir": "ascending",
                      "days": sorted({q_["d"] for q_ in passes if q_["s"] == "s1"}), "model_input": True},
               "nisar": {"name": "NISAR", "time": "06:00", "kind": "L-band SAR", "dir": "descending",
                         "days": [int(x) for x in nisar_days], "model_input": False, "answer_update": p["nisar_update"],
                         "note": f"simulated accuracy {p['nisar_acc']:.2f}, assumed {p['nisar_assumed']:.2f} up to 70 cm plant height, "
                                 f"lower above; "
                                 f"{'updates the forecast as an answer' if p['nisar_update'] else 'shown only'}"}}

    grid = lambda a, r=1: np.round(a.reshape(T, n).T, r)  # noqa: E731
    nan_to_none = lambda a: [[None if not np.isfinite(v) else float(v) for v in row] for row in a]  # noqa: E731
    prog.stage(4)
    rid = uuid.uuid4().hex[:12]
    _runs[rid] = {"p": p, "inp": inp, "sim": sim, "X": X, "H": H, "nisar_said": nisar_said, "cfg": cfg, "no_log": ~logs,
                  "nisar_acc_assumed": acc_assumed,
                  "area": np.array([f_["area_m2"] for f_ in lay["fields"]]),
                  "wet": {"truth": H > 0, "model": pw.reshape(T, n).T > 0.5, "satellite": mh.satellite_wet(X, kp.CH, nisar_said)},
                  "season": season, "truth_fields": np.ones(n, bool)}
    logger_eod = ps.end_of_day_depth(sim["logger"]).astype(np.float64)
    while len(_runs) > 8:
        _runs.popitem(last=False)
    sp_eff = sim["sp_eff"]
    th_med = np.median(th, 0).reshape(T, n, 4)[-1]
    sp_model = np.median(ps.sp_effective(*(th.reshape(K, T, n, 4)[:, -1, :, i] for i in range(4)),
                                         sim["tp_abs"][None, :], inp, dt=0.05), 0)
    return {
        "run": rid, "params": p, "n": n, "T": T, "layout": lay, "mode": "simulation", "planted": season.astype(int).tolist(),
        "start_date": str((inp["day0"] + np.timedelta64(int(inp["s0"]), "D")).date()),
        "et0": np.round(inp["et0"][inp["s0"]:inp["s0"] + T], 2).tolist(),
        "rain": np.round(inp["rain"][inp["s0"]:inp["s0"] + T], 1).tolist(),
        "s2_days": [int(x) for x in inp["s2_days"]],
        "fields": [{"ftype": ["conventional", "AWD", "top-up"][int(sim["ftype"][i])], "logs": bool(logs[i]),
                    "tp": round(float(sim["tp"][i]), 1), "sp_true": round(float(sp_eff[i]), 2),
                    "sp_model": round(float(sp_model[i]), 2), "theta_model": np.round(th_med[i], 4).tolist(),
                    "irr_total": round(float(sim["irr_true_mm"][i]), 0), "area_m2": lay["fields"][i]["area_m2"],
                    "id": lay["fields"][i]["id"], "tw": lay["fields"][i]["tw"]} for i in range(n)],
        # water-level loggers at the site's logger positions: the simulated AWD-pipe reading (3 mm noise,
        # -250..+150 mm pipe range), end of day
        "loggers": [{**lg, "reading": nan_to_none(np.round(np.where(season[lg["field"]], logger_eod[lg["field"]], np.nan), 1)[None, :])[0]}
                    for lg in lay["loggers"]],
        "truth": nan_to_none(np.round(np.where(season, H, np.nan), 1)),
        "irr": np.round(sim["irr_mm"], 0).tolist(),
        "lai": np.round(np.where(season, crop["lai"], 0.0), 2).tolist(),
        "crop": _crop_payload(crop, ripen, season), "sensing_tables": _sensing_tables(inp, cfg.nisar_acc),
        "logged": (X[:, :, kp.CH["irr_logged"]] > 0).astype(int).tolist(),
        "belief": {"p_wet": nan_to_none(grid(np.where(in_season, pw, np.nan), 3)),
                   "h": nan_to_none(grid(np.where(in_season, expect, np.nan))),
                   "q05": nan_to_none(grid(np.where(in_season, q05, np.nan))),
                   "q50": nan_to_none(grid(np.where(in_season, q50, np.nan))),
                   "q95": nan_to_none(grid(np.where(in_season, q95, np.nan)))},
        "dry_pred": nan_to_none(grid(D_pred)), "dry_true": nan_to_none(grid(D_true)),
        "alert_days": ALERT_DAYS, "passes": passes, "sensors": sensors, "nisar_update": nisar_stats,
        "seconds": round(time.time() - t_start, 1),
    }


DEPTH_LABELS = ["no water", "a little (0-20 mm)", "some (20-50 mm)", "a lot (> 50 mm)"]


@_locked
def ask_farmer(rid, f, d, kind="yesno", answer=None):
    """A farmer's answer about field f at the end of day d and the belief before and after it. The answer is
    simulated from the truth with the error rates of Experiment 13, or given (`answer`: 0/1 for yesno, a depth
    category 0-3 for depth), as when someone taps it on the farmer's phone view. kind "none" asks nothing and
    returns the advice as it stands."""
    r = _runs.get(rid)
    if r is None:
        raise KeyError("run expired; run the season again")
    if r.get("mode") == "replay":
        raise ValueError("replay mode: farmers' real answers come from the app log, not from simulated questions")
    sim, X, H, inp = r["sim"], r["X"], r["H"], r["inp"]
    ps.set_temperature(inp["tmean"])
    T = X.shape[1]
    d = int(min(max(d, 0), T - 1))
    if not X[f, d, kp.CH["season"]] > 0:
        raise ValueError("field not transplanted yet on this day")
    th, st = kp.predict_v1(model(r["p"]["model"]), X[f:f + 1], sim["static"][f:f + 1], d + 1, n_samples=K_ASK, seed=f)
    rng = np.random.default_rng(1000 * f + d)
    hs = np.maximum(kp.sample_depth(st, d, K_ASK, rng), ps.PIPE_MIN)  # (K_ASK, 1)
    truth = float(H[f, d])
    if kind == "none":
        answer, label = None, None
    elif answer is not None:
        answer = int(answer)
        if kind == "depth" and not 0 <= answer <= 3 or kind == "yesno" and answer not in (0, 1):
            raise ValueError("answer out of range")
        label = DEPTH_LABELS[answer] if kind == "depth" else ("yes, there is water" if answer else "no water")
    elif kind == "depth":
        cat = int(kp.depth_category(truth))
        if rng.random() > 0.95:  # wet / dry mistake
            cat = 0 if cat > 0 else 1
        elif cat > 0 and rng.random() > 0.8:
            cat = int(np.clip(cat + rng.choice([-1, 1]), 1, 3))
        answer = cat
        label = DEPTH_LABELS[cat]
    else:
        answer = int((truth > 0) != (rng.random() > 0.95))
        label = "yes, there is water" if answer else "no water"
    wts = np.ones_like(hs) if kind == "none" else kp.answer_likelihood(hs, np.array([answer]), kind)
    base = np.ones_like(hs)   # "before" includes the next morning's NISAR pass, as the app's belief does
    said = int(r["nisar_said"][f, d + 1]) if d + 1 < T else -1
    nisar_used = r["p"]["nisar_update"] and said >= 0
    if nisar_used:
        base = kp.answer_likelihood(hs, np.array([said]), "yesno", p_correct=float(r["nisar_acc_assumed"][f, d + 1]))
    t0 = inp["s0"] + d + 1.0
    D = _forecast(th, hs, t0, sim["tp_abs"][f:f + 1][None, :], inp)

    def summary(w):
        qh = kp.weighted_quantiles(hs, w, [0.05, 0.5, 0.95])[:, 0]
        wet = hs[:, 0] > 0
        pond = float(np.sum(w[:, 0] * wet) / np.sum(w[:, 0]))
        qd = kp.weighted_quantiles(D, w, [0.05, 0.5, 0.95])[:, 0] if pond > 0.02 else [0.0, 0.0, 0.0]
        q8 = kp.weighted_quantiles(D, w, [0.1, 0.5, 0.9])[:, 0] if pond > 0.02 else [0.0, 0.0, 0.0]
        p_dry1 = float(np.sum(w[:, 0] * (D[:, 0] <= ALERT_DAYS)) / np.sum(w[:, 0]))
        return {"p_wet": round(pond, 3), "h": [round(float(x), 1) for x in qh],
                "dry_in": [round(float(x), 2) for x in qd], "dry_80": [round(float(x), 2) for x in q8],
                "p_dry_tomorrow": round(p_dry1, 3)}

    t_dry = kp.forecast_dry(sim["theta"][f], truth, t0, sim["tp_abs"][f], inp, et_f=sim["et_f"][f], g=sim["g"][f],
                            max_days=MAX_DAYS, dt=FC_DT)
    return {"field": f, "day": d, "kind": kind, "answer": label, "answer_code": answer, "truth_h": round(truth, 1),
            "truth_dry_in": round(float(t_dry), 2), "before": summary(base), "after": summary(base * wts),
            "nisar": (["no water", "water"][said] if nisar_used else None)}


# ── replay mode: the same model on campaign data (app/campaign.py) ───────────────
def _observed_next_dry(L6, logged, rain, T, jump=10.0, max_gap=4):
    """Realised days until the water was gone after the end of each day, from a logger's 6-hourly readings,
    as the forecast defines it (no irrigation): censored (NaN) when a rise of more than `jump` mm (irrigation or
    rain), a logged irrigation or a gap longer than a day comes first; 0 when already not ponded; capped at
    the forecast horizon."""
    D = np.full(T, np.nan)
    for d in range(T):
        k0 = (d + 1) * campaign.PER_DAY
        if k0 >= len(L6) or not np.isfinite(L6[k0]):
            continue
        if L6[k0] <= 0:
            D[d] = 0.0
            continue
        last, gap = L6[k0], 0
        for k in range(k0 + 1, min(len(L6), k0 + int(MAX_DAYS * campaign.PER_DAY) + 1)):
            day_k = (k - 1) // campaign.PER_DAY
            if day_k < T and (logged[day_k] or rain[day_k] >= 1.0):
                break
            v = L6[k]
            if not np.isfinite(v):
                gap += 1
                if gap > max_gap:
                    break
                continue
            gap = 0
            if v - last > jump:
                break
            if v <= 0:
                D[d] = (k - k0) / campaign.PER_DAY
                break
            last = v
        else:
            D[d] = MAX_DAYS
    return D


@_locked
def run_replay(name, params, base_inp, token=None):
    """Run KGML-Paddy on a campaign's observations. Truth is the loggers; other fields show the model only."""
    t_start = time.time()
    prog = _Progress(token, REPLAY_STAGES)
    p = clean_params(params)
    metas = {c_["name"]: c_ for c_ in campaign.list_campaigns()}
    if name not in metas:
        raise KeyError(f"unknown campaign {name!r}")
    lay = sites.load_site(metas[name]["site"])
    c = campaign.load(name, lay)
    n, T, tp = c["n"], c["T"], c["tp"]
    ps.set_temperature(c["tmean"])          # the campaign's own temperature drives the crop (day 0 = season start)
    p.update({"site": metas[name]["site"], "n_fields": n, "campaign": name})
    inp = {**{k: base_inp[k] for k in ("lai_lo", "rec", "spec", "s1_bins")}, "et0": c["et0"], "rain": c["rain"],
           "day0": c["start"], "s0": 0, "s_end": T, "t_mid": c["t_mid"],
           "s2_days": np.flatnonzero((c["sat"]["s2"] >= 0).any(0))}
    days = np.arange(T)
    season = days[None, :] + 1 >= tp[:, None]
    lai = campaign.daily_lai(c, ps.lai_curve)
    ch = kp.CH
    X = np.zeros((n, T, len(ps.CHANNELS)), np.float32)
    X[:, :, ch["season"]] = season
    X[:, :, ch["days_since_tp"]] = np.clip(days[None, :] - tp[:, None], 0, None) / 100 * season
    X[:, :, ch["et0"]] = c["et0"][:T][None, :] / 10
    X[:, :, ch["rain"]] = c["rain"][:T][None, :] / 20
    X[:, :, ch["lai_obs"]] = np.clip(lai, 0, 6) / 5 * season
    for s_ in ("s2", "s1"):
        X[:, :, ch[f"{s_}_wet"]] = (c["sat"][s_] == 1) & season
        X[:, :, ch[f"{s_}_dry"]] = (c["sat"][s_] == 0) & season
    X[:, :, ch["irr_logged"]] = c["logged"]
    X[:, :, ch["depth_logged"]] = c["irr_mm"] / 100
    X[:, :, ch["gone"]] = c["gone"]
    static = np.zeros((n, ps.N_STATIC), np.float32)   # no field attributes yet: the model's average field
    nisar_said = np.where(season, c["sat"]["nisar"], -1).astype(np.int8)
    crop, ripen = _crop(tp, np.ones(n), T, 0)
    acc_assumed = _nisar_grid(p["nisar_assumed"], crop, season)

    prog.stage(1)
    b = _belief(X, static, season, tp, inp, 0, nisar_said, p["nisar_update"], acc_assumed, p["model"], prog)
    prog.stage(3)
    fld, day, th, pw, pw_model, in_season, upd, iu = (b[k] for k in ("fld", "day", "th", "pw", "pw_model", "in_season", "upd", "iu"))

    # truth: the loggers (6-hourly grid; a field with several loggers takes their mean)
    L6 = np.full((n, T * campaign.PER_DAY + 1), np.nan)
    by_field = {}
    for lg in lay["loggers"]:
        by_field.setdefault(lg["field"], []).append(c["logger6"][lg["id"]])
    for f, arrs in by_field.items():
        with np.errstate(all="ignore"):
            L6[f] = np.nanmean(np.vstack(arrs), 0) if len(arrs) > 1 else arrs[0]
    H = np.clip(L6[:, campaign.PER_DAY::campaign.PER_DAY][:, :T], ps.PIPE_MIN, ps.PIPE_MAX)
    H = np.where(season, H, np.nan)
    D_obs = np.full((n, T), np.nan)
    for f in by_field:
        D_obs[f] = _observed_next_dry(L6[f], c["logged"][f], c["rain"], T)
    D_obs = np.where(season, D_obs, np.nan)

    # satellite passes, checked against the model's prior (N3) and, where there is a logger, the logger
    prior = np.full((n, T), 0.5)
    prior[:, 1:] = pw.reshape(T, n).T[:, :-1]
    hour_q = {"nisar": 1, "s2": 2, "s1": 3}
    passes = []
    for s_ in ("nisar", "s2", "s1"):
        pf, pd_ = np.nonzero((c["sat"][s_] >= 0) & season)
        if not len(pf):
            continue
        said = c["sat"][s_][pf, pd_] == 1
        if s_ == "nisar":
            rec = spec = acc_assumed[pf, pd_]
        else:
            rec, spec = kp.pass_rates(inp, s_, lai[pf, pd_])
        pr = prior[pf, pd_]
        fused = kp.fuse_pass(pr, said, rec, spec)
        flag = (said & (pr < 0.1)) | (~said & (pr > 0.9))
        lv = L6[pf, pd_ * campaign.PER_DAY + hour_q[s_]]
        for a_, d_, w_, l_, g_, h_, k_ in zip(pf, pd_, said, lv, pr, fused, flag):
            q_ = {"s": s_, "f": int(a_), "d": int(d_), "wet": bool(w_), "truth": (bool(l_ > 0) if np.isfinite(l_) else None),
                  "prior": round(float(g_), 3), "fused": round(float(h_), 3), "flag": bool(k_)}
            if s_ == "nisar":
                q_["update"] = bool(d_ >= 1 and upd[(d_ - 1) * n + a_])
            passes.append(q_)
    passes.sort(key=lambda q_: (q_["d"], q_["f"]))

    # scores against the loggers only
    Hf, Df = H.T.reshape(-1), D_obs.T.reshape(-1)
    ok = in_season & np.isfinite(Hf)
    okd = ok & np.isfinite(Df)
    pred_d = np.nan_to_num(b["D_pred"], nan=0.0)
    stats = {"logger_field_days": int(ok.sum()), "ponded_bacc": _bacc(Hf[ok] > 0, pw[ok] > 0.5),
             "next_dry_err_days": float(np.median(np.abs(pred_d[okd] - Df[okd]))) if okd.any() else None,
             "forecasts_scored": int(okd.sum()), "forecasts_censored": int((ok & ~np.isfinite(Df)).sum())}
    nisar_stats = {"enabled": p["nisar_update"], "n": int(len(iu)), "assumed": p["nisar_assumed"],
                   "true": c["nisar_accuracy"], "scored_on": "logger fields"}
    evp = iu[np.isfinite(Hf[iu])] if len(iu) else iu
    evd = evp[np.isfinite(Df[evp])] if len(evp) else evp
    if len(evp):
        nisar_stats.update({"n_scored": int(len(evp)), "ponded_bacc_before": _bacc(Hf[evp] > 0, pw_model[evp] > 0.5),
                            "ponded_bacc_after": _bacc(Hf[evp] > 0, pw[evp] > 0.5)})
    if len(evd):
        nisar_stats.update({"dry_err_before": float(np.median(np.abs(np.nan_to_num(b["D_before"][evd], nan=0.0) - Df[evd]))),
                            "dry_err_after": float(np.median(np.abs(pred_d[evd] - Df[evd])))})

    # data health per logger (coverage of the in-season 6-hourly grid, last reading)
    health = []
    for lg in lay["loggers"]:
        arr = c["logger6"][lg["id"]]
        f = lg["field"]
        grid_in = np.repeat(season[f], campaign.PER_DAY)
        got = np.isfinite(arr[:T * campaign.PER_DAY]) & grid_in
        last = int(np.flatnonzero(np.isfinite(arr))[-1] // campaign.PER_DAY) if np.isfinite(arr).any() else None
        health.append({"id": lg["id"], "field": f, "coverage": round(float(got.sum() / max(1, grid_in.sum())), 3),
                       "last_day": last, "stopped": last is not None and last < T - 3})

    # each logger field's S&P from its own recession days (as Experiment 9), and the model's estimate
    rr = dict(zip(("h", "tau", "r"), ps.recession_rates(L6[:, :T * campaign.PER_DAY + 1], tp, inp)))
    sp_log = kp.logger_sp_eff({**rr, "tp_abs": tp}, inp)
    th_last = th.reshape(K, T, n, 4)[:, -1]
    sp_model = np.median(ps.sp_effective(*(th_last[..., i] for i in range(4)), tp[None, :], inp, dt=0.05), 0)

    grid = lambda a, r=1: np.round(a.reshape(T, n).T, r)  # noqa: E731
    nan_to_none = lambda a: [[None if not np.isfinite(v) else float(v) for v in row] for row in a]  # noqa: E731
    q50_ft = np.where(in_season, b["q50"], np.nan).reshape(T, n).T
    qcr = qc.check(c, lay, season, q50_ft, passes, pipe=(ps.PIPE_MIN, ps.PIPE_MAX))
    prog.stage(4)
    rid = uuid.uuid4().hex[:12]
    _runs[rid] = {"mode": "replay", "qc": qcr, "label": c["meta"].get("label", name), "season": season,
                  "area": np.array([f_["area_m2"] for f_ in lay["fields"]]),
                  "wet": {"truth": np.nan_to_num(H, nan=-1.0) > 0, "model": pw.reshape(T, n).T > 0.5,
                          "satellite": mh.satellite_wet(X, kp.CH, nisar_said)},
                  "truth_fields": np.isfinite(H).any(1)}
    while len(_runs) > 8:
        _runs.popitem(last=False)
    has_app = c["logged"].any(1) | c["gone"].any(1)
    meta = c["meta"]
    return {
        "run": rid, "params": p, "n": n, "T": T, "layout": lay, "mode": "replay", "planted": season.astype(int).tolist(),
        "campaign": {"name": name, "label": meta.get("label", name), "synthetic": bool(meta.get("synthetic", False)),
                     "source": meta.get("source", ""), "issues": c["issues"] + list(meta.get("demo_injected_issues", [])),
                     "farmers_using_app": int(has_app.sum()), "logged_irrigations": int(c["logged"].sum()),
                     "satellite_calls": {k: int(((v >= 0) & season).sum()) for k, v in c["sat"].items()},
                     "meters": c["meters"], "logger_health": health},
        "start_date": str(c["start"].date()),
        "et0": np.round(c["et0"][:T], 2).tolist(), "rain": np.round(c["rain"][:T], 1).tolist(),
        "s2_days": [int(x) for x in inp["s2_days"]],
        "fields": [{"ftype": "unknown", "logs": bool(has_app[i]), "tp": round(float(tp[i]), 1),
                    "sp_true": (round(float(sp_log[i]), 2) if np.isfinite(sp_log[i]) else None),
                    "sp_model": round(float(sp_model[i]), 2), "irr_total": round(float(c["irr_mm"][i].sum()), 0),
                    "area_m2": lay["fields"][i]["area_m2"], "id": lay["fields"][i]["id"], "tw": lay["fields"][i]["tw"]}
                   for i in range(n)],
        "loggers": [{**lg, "reading": nan_to_none(np.round(np.where(season[lg["field"]], np.clip(
            c["logger6"][lg["id"]][campaign.PER_DAY::campaign.PER_DAY][:T], ps.PIPE_MIN, ps.PIPE_MAX), np.nan), 1)[None, :])[0]}
                    for lg in lay["loggers"]],
        "truth": nan_to_none(np.round(H, 1)),
        "irr": np.round(c["irr_mm"], 0).tolist(), "logged": c["logged"].astype(int).tolist(),
        "lai": np.round(np.where(season, lai, 0.0), 2).tolist(),
        "crop": _crop_payload(crop, ripen, season), "sensing_tables": _sensing_tables(inp, p["nisar_assumed"]),
        "belief": {"p_wet": nan_to_none(grid(np.where(in_season, pw, np.nan), 3)),
                   "h": nan_to_none(grid(np.where(in_season, b["expect"], np.nan))),
                   "q05": nan_to_none(grid(np.where(in_season, b["q05"], np.nan))),
                   "q50": nan_to_none(grid(np.where(in_season, b["q50"], np.nan))),
                   "q95": nan_to_none(grid(np.where(in_season, b["q95"], np.nan)))},
        "dry_pred": nan_to_none(grid(b["D_pred"])), "dry_true": nan_to_none(np.round(D_obs, 2)),
        "alert_days": ALERT_DAYS, "passes": passes, "nisar_update": nisar_stats, "replay_stats": stats, "qc": qcr,
        "sensors": {"s2": {"name": "Sentinel-2", "time": "10:30", "kind": "optical", "dir": "descending", "model_input": True,
                           "days": sorted({q_["d"] for q_ in passes if q_["s"] == "s2"})},
                    "s1": {"name": "Sentinel-1", "time": "18:00", "kind": "C-band SAR", "dir": "ascending", "model_input": True,
                           "days": sorted({q_["d"] for q_ in passes if q_["s"] == "s1"})},
                    "nisar": {"name": "NISAR", "time": "06:00", "kind": "L-band SAR", "dir": "descending", "model_input": False,
                              "answer_update": p["nisar_update"], "days": sorted({q_["d"] for q_ in passes if q_["s"] == "nisar"})}},
        "seconds": round(time.time() - t_start, 1),
    }


def progress_done(token):
    """Mark a run's progress finished (the server calls this once the result is ready to send)."""
    if token and token in PROGRESS:
        PROGRESS[token].update(frac=1.0, done=True)


def qc_report(rid, fmt="md", week=None):
    """The quality-control report of a replay run: Markdown (one week, default the last) or the issues as CSV."""
    if rid not in _runs or _runs[rid].get("mode") != "replay":
        raise KeyError("run not found (replay runs only; run the campaign again)")
    r = _runs[rid]
    if fmt == "csv":
        return qc.issues_csv(r["qc"])
    if week is not None and not -len(r["qc"]["weeks"]) <= week < len(r["qc"]["weeks"]):
        raise ValueError("no such week")
    return qc.report_markdown(r["qc"], r["label"], week)


# ── advice against practice (src/awdrt/counterfactual.py) ─────────────────────
CF_ARMS = ("practice", "model", "regional", "et_only", "perfect")
CF_LIMITS = {"wait": (0, 3), "risk": (0.05, 0.5), "lead": (1, 5), "follow": (0.0, 1.0)}
CF_DEFAULTS = {"wait": 0, "risk": 0.5, "lead": 1, "follow": 1.0, "ask": True}


@_locked
def run_counterfactual(rid, params=None):
    """Re-run a simulated season's village under each irrigation policy (same fields, weather and draws)."""
    if rid not in _runs or "sim" not in _runs[rid]:
        raise KeyError("run not found (simulation runs only; run the season again)")
    r = _runs[rid]
    q = dict(CF_DEFAULTS)
    q.update({k: v for k, v in (params or {}).items() if k in CF_DEFAULTS})
    for k, (lo, hi) in CF_LIMITS.items():
        q[k] = min(max(float(q[k]), lo), hi)
    q["wait"], q["lead"] = int(round(q["wait"])), float(round(q["lead"]))
    q["ask"] = bool(q["ask"]) and q["ask"] not in ("0", "false", "False")
    t_start = time.time()
    sim, inp, area = r["sim"], r["inp"], r["area"]
    ps.set_temperature(inp["tmean"])
    n = len(area)
    with torch.no_grad():
        arms = cf.run(sim, r["cfg"], inp, CF_ARMS, seed=r["p"]["seed"], wait=q["wait"], follow=q["follow"], model=model(r["p"]["model"]),
                      n_samples=K, risk=q["risk"], lead=q["lead"], no_log=r["no_log"], ask=q["ask"])
    ftype = np.array(["conventional", "AWD", "top-up"])[sim["ftype"]]
    out = {}
    for name, A in arms.items():
        o = cf.outcomes(A, area)
        groups = {}
        for g in ("conventional", "AWD", "top-up"):
            m = ftype == g
            if m.any():
                groups[g] = {"fields": int(m.sum()), "water_mm": round(float(o["water_mm"][m].mean()), 0),
                             "stress_days": round(float(o["stress_days"][m].mean()), 1), "irrigations": round(float(o["irrigations"][m].mean()), 1)}
        cum = np.cumsum((A["irr_mm"] * area[:, None] / 1000).sum(0))
        reg = mh.regime(mh.drainage_periods(np.nan_to_num(A["H"], nan=-1.0) > 0, A["season"]))
        kg_ha = mh.emissions(reg, A["season"].sum(1))
        out[name] = {
            "total_m3": round(float(o["water_m3"].sum()), 0), "water_mm": round(float(o["water_mm"].mean()), 0),
            "irrigations": round(float(o["irrigations"].mean()), 2), "ponded_days": round(float(o["ponded_days"].mean()), 1),
            "dry_days": round(float(o["dry_days"].mean()), 1), "stress_days": round(float(o["stress_days"].mean()), 2),
            "asked": round(float(A["asked"].sum(1).mean()), 1),
            "methane": {"kg_ha": round(float(kg_ha.mean()), 1), "total_kg": round(float(kg_ha @ (area / 1e4)), 0),
                        "counts": {g: int((reg == g).sum()) for g in mh.REGIMES[:3]}, "field_kg_ha": np.round(kg_ha, 1).tolist()},
            "by_type": groups, "cum_m3": np.round(cum, 0).tolist(),
            "field": {"water_mm": np.round(o["water_mm"], 0).tolist(), "stress_days": o["stress_days"].astype(int).tolist(),
                      "irrigations": o["irrigations"].astype(int).tolist()},
            "H": [[None if not np.isfinite(v) else round(float(v), 0) for v in row] for row in np.clip(A["H"], ps.PIPE_MIN, ps.PIPE_MAX)],
            "irr": np.round(A["irr_mm"], 0).tolist(), "asked_days": [np.flatnonzero(row).tolist() for row in A["asked"]],
        }
    return {"run": rid, "settings": q, "arms": out, "advised": arms["model"]["advised"].astype(int).tolist(),
            "ftype": ftype.tolist(), "n": n, "stress_mm": cf.STRESS_MM, "seconds": round(time.time() - t_start, 1)}


# ── methane (src/awdrt/methane.py): IPCC Tier 1 from each field's water regime ──
MH_DEFAULTS = {"min_dry": mh.MIN_DRY_DAYS, "regional": "aggregated"}


def run_methane(rid, params=None):
    """Each field's water regime and Tier 1 methane, from the truth (or loggers), the model's belief, the
    satellite calls alone, and one regional default for every field."""
    if rid not in _runs or "wet" not in _runs[rid]:
        raise KeyError("run not found (run the season again)")
    r = _runs[rid]
    q = dict(MH_DEFAULTS)
    q.update({k: v for k, v in (params or {}).items() if k in MH_DEFAULTS})
    q["min_dry"] = int(min(max(int(q["min_dry"]), 1), 7))
    if q["regional"] not in mh.SF_W:
        q["regional"] = "aggregated"
    season, area_ha = r["season"], r["area"] / 1e4
    tf = r["truth_fields"]
    n = len(area_ha)
    out = {}
    for k, wet in r["wet"].items():
        per = mh.drainage_periods(np.where(season, wet, False), season, q["min_dry"])
        out[k] = mh.regime(per)
        out[k + "_periods"] = per
    out["regional"] = np.full(n, q["regional"])
    days = season.sum(1)
    sources = {}
    for k in ("truth", "model", "satellite", "regional"):
        reg = out[k]
        sel = tf if k == "truth" else np.ones(n, bool)
        kg = mh.emissions(reg, days)
        c, lo, hi = mh.uncertainty(reg[sel], days[sel], area_ha[sel])
        src = {"regime": reg.tolist(), "kg_ha": [round(float(v), 1) if s_ else None for v, s_ in zip(kg, sel)],
               "periods": out[k + "_periods"].tolist() if k + "_periods" in out else None,
               "total_kg": round(c, 0), "total_lo": round(lo, 0), "total_hi": round(hi, 0),
               "counts": {g: int((reg[sel] == g).sum()) for g in mh.REGIMES}}
        if k != "truth":     # against the truth (all fields in simulation; logger fields in replay)
            t_reg, t_kg = out["truth"][tf], mh.emissions(out["truth"][tf], days[tf])
            m_kg = kg[tf]
            src["vs_truth"] = {"fields": int(tf.sum()),
                               "regime_agree": round(float((reg[tf] == t_reg).mean()), 3) if k != "regional" else None,
                               # mean recall over the true regimes present (a source that calls every field "multiple"
                               # scores 0.5 with two regimes present, whatever their shares)
                               "regime_balanced": round(float(np.mean([(reg[tf][t_reg == g] == g).mean() for g in np.unique(t_reg)])), 3)
                               if k != "regional" else None,
                               "cf_recall": round(float((reg[tf][t_reg == "continuous"] == "continuous").mean()), 3)
                               if k != "regional" and (t_reg == "continuous").any() else None,
                               "total_err": round(float((m_kg @ area_ha[tf]) / max(1e-9, t_kg @ area_ha[tf]) - 1), 3),
                               "field_err_kg_ha": round(float(np.mean(np.abs(m_kg - t_kg))), 1),
                               # H5b: fields whose estimate is off by at least half the gap between continuous flooding and AWD
                               "half_gap_share": round(float(np.mean(np.abs(m_kg - t_kg) >= 0.5 * (mh.SF_W["continuous"][0] - mh.SF_W["multiple"][0]) * mh.EF_C["South Asia"][0] * days[tf] - 1e-9)), 3),
                               "confusion": {a: {b: int(((t_reg == a) & (reg[tf] == b)).sum()) for b in mh.REGIMES} for a in mh.REGIMES[:3]}}
        sources[k] = src
    return {"run": rid, "settings": q, "sources": sources, "days": days.tolist(), "truth_fields": tf.astype(int).tolist(),
            "factors": {"ef_c": mh.EF_C["South Asia"], "sf_w": mh.SF_W, "sf_p": mh.SF_P, "gwp100": mh.GWP100,
                        "names": mh.REGIME_NAMES}}
