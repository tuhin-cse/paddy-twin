"""Paddy simulator v2: many Boro fields at once, with the physics and observations KGML-Paddy needs.

Differences from the Experiment 4 twin (scripts/04_field_leakage_twin.py):

* S&P depends on depth and season: SP(h, tau) = a (1 + w tau/80) exp(-c tau) + b h
  (base rate, water table falls, puddled layer seals; Darcy flow through the puddled layer
  and bund seepage in b). Same form as Experiment 8.
* Water below the surface: storage S (mm of water) is ponded depth when S > 0 and
  S / Sy (specific yield) below the surface, so a field drains to -15 cm within hours to a
  day, as loggers show (median 19 h, Remote Sens. 18:2136).
* Lateral exchange between adjacent plots: fields come in pairs (i, i^1) that exchange water
  at k_lat (max(h_i, 0) - max(h_j, 0)).
* Farmer types: conventional (re-irrigate 0.5-3 days after the field dries), AWD
  (re-irrigate at -15 cm) and top-up (re-irrigate while 5-20 mm still stands). Each event is
  logged with a per-farmer probability; logged depth carries a per-field pump-rating bias.
  Some farmers also report "no water this morning".
* Observations on a daily grid: Sentinel-2 on the real calendar with per-field cloud gaps and
  LAI-dependent errors from Experiment 1 (no hard LAI cut-off: the model sees LAI and learns
  how far to trust each pass), Sentinel-1 every 6 days with the accuracy measured in 03b,
  optional NISAR, satellite LAI with noise, and 6-hourly water-level loggers (pipe range
  -250 to +150 mm, 3 mm noise).
* Field attributes: 8 static features, the first 4 explaining a chosen share of the
  variance of log a and log b (unknown in reality; Experiment 4e).

`simulate(cfg, n, seed)` returns a dict of numpy arrays; `WORLDS` holds the pre-training
world and a "reality" with the Experiment 8 mismatches. Weather is the ERA5 Boro season
2023-24 of the Jamalpur window (results/era5_daily_jamalpur.csv).
"""

import json
import pathlib
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
RES = ROOT / "results"

DT = 0.05                      # days per sub-step
PER_DAY = 4                    # logger readings per day (00, 06, 12, 18 h)
SEASON_START = pd.Timestamp("2024-01-15")
SEASON_END = pd.Timestamp("2024-04-15")
T_MID = pd.Timestamp("2024-03-01")  # date of the effective-S&P truth (about LAI 2)
PIPE_MIN, PIPE_MAX = -250.0, 150.0

CHANNELS = ["season", "days_since_tp", "et0", "rain", "lai_obs", "s2_wet", "s2_dry",
            "s1_wet", "s1_dry", "nisar_wet", "nisar_dry", "irr_logged", "depth_logged", "gone"]
N_STATIC = 8


@dataclass(frozen=True)
class SimConfig:
    # physics
    a_median: float = 5.0          # base S&P, mm/d (lognormal)
    a_sigma: float = 0.6
    Lp: tuple = (100.0, 400.0)     # puddled-layer thickness, mm (b = a/Lp + bund seepage)
    s_lat_max: float = 0.03        # bund seepage to drains, 1/d
    c_max: float = 0.008           # sealing, 1/d
    w_max: float = 0.3             # water-table effect over 80 days
    k_lat_max: float = 0.02        # exchange with the paired plot, 1/d
    Sy: tuple = (0.03, 0.10)       # specific yield below the surface
    et0_noise: float = 0.05        # per-field ET0 factor sd
    # farmers
    p_topup: float = 0.15
    p_awd: float = 0.2
    h0_median: tuple = (40.0, 70.0)
    pump_bias_sigma: float = 0.10  # logged depth = true depth x lognormal(log median, sigma)
    pump_bias_median: float = 1.0  # > 1: every pump's nominal rating overstates the depth
    p_log: tuple = (0.8, 1.0)      # per-farmer probability of logging an irrigation
    p_gone_reporter: float = 0.0   # share of farmers who report "no water this morning"
    # observations
    s2_rec_shift: float = 0.0      # added to Experiment 1 recall
    s2_spec_shift: float = 0.0     # added to Experiment 1 specificity
    s1_acc_shift: float = 0.0
    nisar: bool = False
    nisar_acc: float = 0.8
    nisar_revisit: int = 12        # days between NISAR passes (about 6 with ascending + descending)
    lai_noise: float = 0.3
    logger_noise: float = 3.0
    attr_r2: float = 0.3           # share of log a / log b variance the attributes explain


WORLDS = {
    # what the model is pre-trained on
    "prior": SimConfig(),
    # "reality": the Experiment 8 strong + observation mismatches, and a shifted prior
    "reality": SimConfig(a_median=6.0, a_sigma=0.7, Lp=(60.0, 200.0), s_lat_max=0.05, c_max=0.012,
                         w_max=0.5, k_lat_max=0.04, p_topup=0.3, pump_bias_sigma=0.25,
                         p_log=(0.6, 1.0), s2_rec_shift=-0.10, s2_spec_shift=-0.05, s1_acc_shift=-0.05),
}
# "reality" plus systematic errors that loggers can correct: every pump's nominal rating
# overstates the depth by 25 %, and false-wet satellite detections are more frequent.
WORLDS["reality_systematic"] = replace(WORLDS["reality"], pump_bias_median=1.25, s2_spec_shift=-0.15)
# Bounds of the model's parameter space; must cover every world.
BOUNDS = {"c_max": 0.02, "w_max": 1.0}


# ── inputs shared by all fields ────────────────────────────────────────────────

def load_inputs():
    wx = pd.read_csv(RES / "era5_daily_jamalpur.csv", index_col=0, parse_dates=True)
    real = json.loads((RES / "real_validation_metrics.json").read_text())
    season = pd.read_csv(RES / "real_season_curve.csv", index_col="date")
    e1 = json.loads((RES / "simulation_metrics.json").read_text())["E1_background"]
    s1 = json.loads((RES / "s1_wetdry_accuracy.json").read_text())["20m"]
    day0 = wx.index[0]
    s0, s1_ = (SEASON_START - day0).days, (SEASON_END - day0).days
    s2_days = np.array([(pd.Timestamp(d) - day0).days for d in real["dates"]])
    s2_valid = np.array([season.valid_rice_frac.get(d, 0.2) for d in real["dates"]])
    keep = (s2_days >= s0) & (s2_days < s1_)
    lo = np.array(e1["lai_bin_lo"])
    rec = np.array(e1["rt_recall_ponded"])
    spec = 2 * np.array(e1["rt_binary_bacc"]) - rec
    s1_bins = [(b["lai_lo"], b.get("balanced_acc_lodo", 0.58)) for b in s1]
    return dict(et0=wx.et0.to_numpy(), rain=wx.rain.to_numpy(), tmean=wx.tmean.to_numpy(), day0=day0, s0=s0, s_end=s1_,
                s2_days=s2_days[keep] - s0, s2_valid=s2_valid[keep], lai_lo=lo, rec=rec, spec=spec,
                s1_bins=s1_bins, t_mid=(T_MID - day0).days)


def kc_rice(lai):
    """FAO-56 rice Kc under ponding (as awdrt.hydro)."""
    return 1.05 + 0.15 * np.clip(lai / 3.0, 0, 1)


# ── rice growth (phenology) ─────────────────────────────────────────────────────
# "linear": the curve of Experiments 1-26 (LAI 0.5 at transplanting, +0.0375 a day, capped at 5).
# "boro":   growth in thermal time (degree-days above 10 C, from ERA5 daily mean temperature), so the cold of
#           January slows the start and March speeds it up, as in transplanted Boro rice (a ~120-day variety in
#           the field, such as BRRI dhan29): recovery from transplanting shock, active tillering, panicle
#           initiation and booting, heading, grain filling. LAI rises on a logistic curve to about 5.8 near
#           heading (about 1 at 30 days, 3 at maximum tillering, 5.5 by booting), then falls during ripening;
#           plant height goes from 20 to 95 cm; tillers per hill rise to about 22 at maximum tillering and fall
#           to about 15 productive tillers by heading. Each field's growth factor g scales its thermal time.
# The module default stays "linear" so earlier experiments reproduce; Paddy Twin and Experiment 27 use "boro".
PHENOLOGY = "linear"
T_BASE = 10.0
BORO = dict(tt_recover=80.0, lai0=0.4, lai_max=5.8, lai_half=450.0, lai_k=0.008, h0=20.0, h_max=95.0, h_half=550.0,
            h_k=0.007, till0=3.0, till_max=22.0, till_prod=15.0, tt_till_max=520.0, tt_head=1150.0, tt_flower_end=1300.0,
            tt_mature=1700.0, lai_end=0.6)
STAGES = ("recovery", "tillering", "panicle initiation and booting", "heading and flowering", "grain filling", "mature")
_TT = None   # cumulative thermal time (C d) at the start of each day of the current weather series


def set_phenology(kind):
    global PHENOLOGY
    if kind not in ("linear", "boro"):
        raise ValueError(kind)
    PHENOLOGY = kind


def set_temperature(tmean):
    """Daily mean temperature (C) of the weather series that day numbers refer to: ERA5 Jamalpur from its first
    day (load_inputs, the default) or a campaign's weather from its season start (replay)."""
    global _TT
    _TT = np.r_[0.0, np.cumsum(np.clip(np.asarray(tmean, float) - T_BASE, 0, None))]


def _thermal_time():
    if _TT is None:
        set_temperature(pd.read_csv(RES / "era5_daily_jamalpur.csv", index_col=0).tmean.to_numpy())
    return _TT


def _tt_at(t, TT):
    """Cumulative thermal time at (fractional) days t: linear within each day (direct indexing, fast)."""
    t = np.clip(np.asarray(t, float), 0, len(TT) - 1.000001)
    i = t.astype(int)
    return TT[i] + (t - i) * (TT[i + 1] - TT[i])


def thermal_time(t, tp, g=1.0):
    """Degree-days above 10 C from transplanting (absolute day tp) to absolute day t, times the growth factor g."""
    TT = _thermal_time()
    t, tp = np.asarray(t, float), np.asarray(tp, float)
    return np.clip(_tt_at(t, TT) - _tt_at(tp, TT), 0, None) * np.asarray(g, float) * (t >= tp)


def _rise(tt, half, k):
    """A logistic rise from 0 at tt = 0 to 1, normalised so it starts exactly at 0."""
    s0 = 1 / (1 + np.exp(k * half))
    return np.clip((1 / (1 + np.exp(-k * (tt - half))) - s0) / (1 - s0), 0, 1)


def lai_curve(t, tp, g=1.0):
    """Leaf area index on absolute day t of a field transplanted on absolute day tp (growth factor g)."""
    if PHENOLOGY == "linear":
        return np.minimum(0.5 + np.clip(t - tp, 0, None) / 40 * 1.5 * g, 5.0)
    B = BORO
    tt = thermal_time(t, tp, g)
    up = B["lai0"] + (B["lai_max"] - B["lai0"]) * _rise(tt - B["tt_recover"], B["lai_half"], B["lai_k"])
    late = np.clip((tt - B["tt_head"]) / (B["tt_mature"] - B["tt_head"]), 0, 1)
    return up * (1 - (1 - B["lai_end"]) * late)


def crop_state(t, tp, g=1.0):
    """The crop on absolute day t: days after transplanting, thermal time, LAI, height (cm), tillers per hill,
    stage index (STAGES) and canopy cover (1 - exp(-0.5 LAI)). Arrays broadcast."""
    B = BORO
    t, tp = np.asarray(t, float), np.asarray(tp, float)
    tt = thermal_time(t, tp, g) if PHENOLOGY == "boro" else np.clip(t - tp, 0, None) * 10.0 * np.asarray(g, float)
    lai = lai_curve(t, tp, g)
    h = B["h0"] + (B["h_max"] - B["h0"]) * _rise(tt - B["tt_recover"], B["h_half"], B["h_k"])
    rise = B["till0"] + (B["till_max"] - B["till0"]) * _rise(tt - B["tt_recover"], B["tt_till_max"] / 2.2, 0.012)
    fall = np.clip((tt - B["tt_till_max"]) / (B["tt_head"] - B["tt_till_max"]), 0, 1)
    tillers = np.where(tt <= B["tt_till_max"], rise, B["till_max"] - (B["till_max"] - B["till_prod"]) * fall)
    stage = np.select([tt < B["tt_recover"], tt < B["tt_till_max"], tt < B["tt_head"], tt < B["tt_flower_end"],
                       tt < B["tt_mature"]], [0, 1, 2, 3, 4], 5)
    return {"dat": np.clip(t - tp, 0, None), "tt": tt, "lai": lai, "height_cm": h, "tillers": tillers,
            "stage": stage, "cover": 1 - np.exp(-0.5 * lai)}


def nisar_accuracy(acc, height_cm):
    """NISAR's wet/dry accuracy under a canopy of the given height: the nominal accuracy up to 70 cm (the
    PALSAR-2 study's limit), falling by 0.15 by 100 cm, never below 0.55. Constant under "linear" phenology,
    as in Experiments 23-26."""
    if PHENOLOGY == "linear":
        return np.broadcast_to(np.asarray(acc, float), np.shape(height_cm)).copy()
    return np.clip(acc - 0.15 * np.clip((np.asarray(height_cm, float) - 70.0) / 30.0, 0, None), 0.55, 1.0)


def sp_rate(a, b, c, w, h, tau):
    """S&P (mm/d) at ponded depth h (mm) and tau days after transplanting."""
    tau = np.clip(tau, 0, None)
    return a * (1 + w * tau / 80) * np.exp(-c * tau) + b * np.clip(h, 0, None)


def sp_effective(a, b, c, w, tp, inp, h_start=50.0, dt=0.02, max_days=40):
    """What a logger shows: effective S&P of a 50 mm irrigation on T_MID (vectorised).

    SP_eff = (h_start + rain) / D - mean(Kc ET0) over the D days the water lasts, with the
    field's own S&P (lateral exchange excluded, since it depends on the neighbour). Arrays
    broadcast, so posterior samples (k, n) work as well as truths (n,).
    """
    a, b, c, w, tp = np.broadcast_arrays(*(np.asarray(x, float) for x in (a, b, c, w, tp)))
    h = np.full(a.shape, h_start)
    done = np.zeros(a.shape, bool)
    D = np.full(a.shape, float(max_days))
    rain_acc = np.zeros(a.shape)
    et_acc = np.zeros(a.shape)
    t0 = inp["t_mid"]
    for k in range(int(max_days / dt)):
        t = t0 + k * dt
        d = int(t)
        etc = kc_rice(lai_curve(t, tp)) * inp["et0"][d]
        live = ~done
        h = h + live * (inp["rain"][d] - etc - sp_rate(a, b, c, w, h, t - tp)) * dt
        rain_acc += live * inp["rain"][d] * dt
        et_acc += live * etc * dt
        newly = live & (h <= 0)
        D = np.where(newly, (k + 1) * dt, D)
        done |= newly
        if done.all():
            break
    return (h_start + rain_acc) / D - et_acc / D


# ── the simulator ──────────────────────────────────────────────────────────────

def _detect(rng, truth, lai, rec_bins, spec_bins, lo):
    b = np.clip(np.searchsorted(lo, lai, side="right") - 1, 0, len(lo) - 1)
    rec, spec = np.clip(rec_bins[b], 0.5, 1), np.clip(spec_bins[b], 0.5, 1)
    u = rng.random(truth.shape)
    return np.where(truth, u < rec, u > spec)


def simulate(cfg: SimConfig, n: int, seed: int, inp=None):
    """Simulate n fields (n even: fields 2k and 2k+1 are adjacent plots)."""
    assert n % 2 == 0
    inp = inp or load_inputs()
    rng = np.random.default_rng(seed)
    s0, T = inp["s0"], inp["s_end"] - inp["s0"]
    et0_d, rain_d = inp["et0"], inp["rain"]

    # per-field physics
    a = np.clip(rng.lognormal(np.log(cfg.a_median), cfg.a_sigma, n), 0.5, 30)
    b = a / rng.uniform(*cfg.Lp, n) + rng.uniform(0, cfg.s_lat_max, n)
    c = rng.uniform(0, cfg.c_max, n)
    w = rng.uniform(0, cfg.w_max, n)
    k_lat = np.repeat(rng.uniform(0, cfg.k_lat_max, n // 2), 2)
    Sy = rng.uniform(*cfg.Sy, n)
    et_f = 1 + cfg.et0_noise * rng.standard_normal(n)
    g = rng.uniform(0.8, 1.2, n)                                   # crop growth rate
    tp = s0 + np.repeat(rng.uniform(0, 20, n // 2), 2) + rng.uniform(0, 3, n)  # neighbours plant together
    nb = np.arange(n) ^ 1

    # farmers
    u = rng.random(n)
    ftype = np.where(u < cfg.p_topup, 2, np.where(u < cfg.p_topup + cfg.p_awd, 1, 0))  # 0 conv, 1 AWD, 2 top-up
    thr = np.select([ftype == 0, ftype == 1, ftype == 2], [0.0, -150.0, rng.uniform(5, 20, n)])
    dlo = np.select([ftype == 0, ftype == 1, ftype == 2], [0.5, 0.0, 0.0])
    dhi = np.select([ftype == 0, ftype == 1, ftype == 2], [3.0, 1.0, 1.0])
    h0_med = rng.uniform(*cfg.h0_median, n)
    pump_bias = rng.lognormal(np.log(cfg.pump_bias_median), cfg.pump_bias_sigma, n)
    p_log = rng.uniform(*cfg.p_log, n)
    reporter = rng.random(n) < cfg.p_gone_reporter

    # integrate
    S = np.zeros(n)
    started = np.zeros(n, bool)
    next_irr = np.full(n, np.inf)
    was_ponded = np.zeros(n, bool)
    steps = int(T / DT)
    rec_every = int(round(1 / (PER_DAY * DT)))
    h6 = np.full((n, T * PER_DAY), np.nan)
    irr_events, gone_events = [], []
    for k in range(steps):
        t = s0 + k * DT
        d = int(t)
        start_now = ~started & (t >= tp)
        if start_now.any():  # puddling flood at transplanting
            S[start_now] = rng.lognormal(np.log(h0_med[start_now]), 0.2)
            started |= start_now
            was_ponded |= start_now
        irr = started & (t >= next_irr)
        if irr.any():
            depth = rng.lognormal(np.log(h0_med[irr]), 0.2)
            S[irr] += depth
            next_irr[irr] = np.inf
            irr_events.append((np.flatnonzero(irr), np.full(irr.sum(), t), depth))
        h = np.where(S > 0, S, S / Sy)
        tau = t - tp
        lai = lai_curve(t, tp, g)
        etc = kc_rice(lai) * et0_d[d] * et_f
        pond = S > 0
        base = a * (1 + w * np.clip(tau, 0, None) / 80) * np.exp(-c * np.clip(tau, 0, None))
        sp = np.where(pond, base + b * h, base)
        hp = np.clip(h, 0, None)
        lat = k_lat * (hp - hp[nb])
        dS = (rain_d[d] - etc - sp - lat) * DT
        S = np.where(started, np.maximum(S + dS, PIPE_MIN * Sy), 0.0)
        h = np.where(S > 0, S, S / Sy)
        # "water gone": ponding ended; reported next morning by some farmers, half the time
        ended = started & was_ponded & (h <= 0)
        if (ended & reporter).any():
            rep = ended & reporter & (rng.random(n) < 0.5)
            gone_events.append((np.flatnonzero(rep), np.floor(t) + 1 + rng.uniform(0, 0.3, rep.sum())))
        was_ponded = started & (h > 0)
        trig = started & np.isinf(next_irr) & (h < thr)
        if trig.any():
            next_irr[trig] = t + rng.uniform(dlo[trig], dhi[trig])
        if k % rec_every == 0:
            h6[:, k // rec_every] = np.where(started, h, np.nan)

    # daily grid, day index 0 = SEASON_START
    days = np.arange(T)
    X = np.zeros((n, T, len(CHANNELS)), np.float32)
    ch = {name: i for i, name in enumerate(CHANNELS)}
    season = (days[None, :] + s0 + 1 >= tp[:, None])  # a day counts once transplanting happened by its end
    X[:, :, ch["season"]] = season
    X[:, :, ch["days_since_tp"]] = np.clip(days[None, :] + s0 - tp[:, None], 0, None) / 100 * season
    X[:, :, ch["et0"]] = et0_d[s0:s0 + T][None, :] * et_f[:, None] / 10
    X[:, :, ch["rain"]] = rain_d[s0:s0 + T][None, :] / 20
    lai_d = lai_curve(days[None, :] + s0 + 0.5, tp[:, None], g[:, None])
    X[:, :, ch["lai_obs"]] = np.clip(lai_d + cfg.lai_noise * rng.standard_normal(lai_d.shape), 0, 6) / 5 * season

    def wet_at(day_idx, q):  # truth at reading q (0..3) of a day
        return h6[:, day_idx * PER_DAY + q] > 0

    # Sentinel-2 (about 10:30): the 12:00 reading
    rec = inp["rec"] + cfg.s2_rec_shift
    spec = inp["spec"] + cfg.s2_spec_shift
    for dd, pv in zip(inp["s2_days"], inp["s2_valid"]):
        ok = season[:, dd] & (rng.random(n) < pv)
        o = _detect(rng, wet_at(dd, 2), lai_d[:, dd], rec, spec, inp["lai_lo"])
        X[:, dd, ch["s2_wet"]] = ok & o
        X[:, dd, ch["s2_dry"]] = ok & ~o
    # Sentinel-1 every 6 days (18:00 reading), symmetric accuracy by LAI bin
    lo1 = np.array([bb[0] for bb in inp["s1_bins"]])
    acc1 = np.array([bb[1] for bb in inp["s1_bins"]]) + cfg.s1_acc_shift
    for dd in range(2, T, 6):
        ok = season[:, dd]
        o = _detect(rng, wet_at(dd, 3), lai_d[:, dd], acc1, acc1, lo1)
        X[:, dd, ch["s1_wet"]] = ok & o
        X[:, dd, ch["s1_dry"]] = ok & ~o
    if cfg.nisar:  # every cfg.nisar_revisit days (06:00 reading)
        for dd in range(5, T, cfg.nisar_revisit):
            ok = season[:, dd]
            truth = wet_at(dd, 1)
            acc = nisar_accuracy(cfg.nisar_acc, crop_state(dd + s0 + 0.25, tp, g)["height_cm"])
            o = np.where(truth, rng.random(n) < acc, rng.random(n) > acc)
            X[:, dd, ch["nisar_wet"]] = ok & o
            X[:, dd, ch["nisar_dry"]] = ok & ~o
    # farmer log
    n_irr = np.zeros(n, int)
    irr_true_mm = np.zeros(n)
    irr_mm = np.zeros((n, T))  # true irrigation depth per day
    for f, t_, dep in irr_events:
        dd = np.clip((t_ - s0).astype(int), 0, T - 1)
        n_irr[f] += 1
        irr_true_mm[f] += dep
        np.add.at(irr_mm, (f, dd), dep)
        logged = rng.random(f.size) < p_log[f]
        np.add.at(X[:, :, ch["irr_logged"]], (f[logged], dd[logged]), 1.0)
        np.add.at(X[:, :, ch["depth_logged"]], (f[logged], dd[logged]), dep[logged] * pump_bias[f[logged]] / 100)
    X[:, :, ch["irr_logged"]] = np.minimum(X[:, :, ch["irr_logged"]], 1)
    for f, t_ in gone_events:
        dd = (t_ - s0).astype(int)
        m = dd < T
        X[f[m], dd[m], ch["gone"]] = 1

    # static field attributes
    za = (np.log(a) - np.log(cfg.a_median)) / cfg.a_sigma
    zb = (np.log(b) - np.log(b).mean()) / np.log(b).std()
    signal = np.stack([za, zb, za + zb, za - zb], 1) / np.array([1, 1, np.sqrt(2), np.sqrt(2)])
    static = np.concatenate([np.sqrt(cfg.attr_r2) * signal + np.sqrt(1 - cfg.attr_r2) * rng.standard_normal((n, 4)),
                             rng.standard_normal((n, N_STATIC - 4))], 1).astype(np.float32)

    logger = np.clip(h6 + cfg.logger_noise * rng.standard_normal(h6.shape), PIPE_MIN, PIPE_MAX)
    return dict(X=X, static=static, theta=np.stack([a, b, c, w], 1), tp=tp - s0, tp_abs=tp,
                sp_eff=sp_effective(a, b, c, w, tp, inp), k_lat=k_lat, Sy=Sy, ftype=ftype,
                n_irr=n_irr, irr_true_mm=irr_true_mm, irr_mm=irr_mm, pump_bias=pump_bias, h6=h6, logger=logger,
                et_f=et_f, g=g,
                farmer=dict(thr=thr, dlo=dlo, dhi=dhi, h0_med=h0_med, p_log=p_log, reporter=reporter))


# ── logger processing (also for the real campaign) ─────────────────────────────

def recession_rates(logger, tp_abs, inp, et_f=None, g=None, min_h=10.0, jump=10.0, max_rain=1.0):
    """Daily S&P observed by a logger on clean recession days.

    A day (00:00 to next 00:00, readings every 6 h) is used when the field stays ponded
    above min_h, no 6-hour step rises by more than `jump` mm (no irrigation; an irrigation adds
    tens of mm, and a tighter threshold would select days where noise happened to fall), and
    rain is below max_rain. The daily drop is the least-squares slope through the 5 readings.
    Returns per field arrays (h_mid, tau, rate) with NaN padding, where
    rate = drop + rain - Kc ET0 (mm/d), i.e. S&P plus any lateral exchange. Kc uses the nominal
    LAI curve and ET0 the regional value, as would be done with real loggers.
    """
    n, R = logger.shape
    T = R // PER_DAY
    s0 = inp["s0"]
    et_f = np.ones(n) if et_f is None else et_f
    g = np.ones(n) if g is None else g
    out_h, out_tau, out_r = (np.full((n, T - 1), np.nan) for _ in range(3))
    for d in range(T - 1):
        seg = logger[:, d * PER_DAY:(d + 1) * PER_DAY + 1]
        ok = np.all(seg > min_h, 1) & np.all(np.diff(seg, axis=1) < jump, 1) & (inp["rain"][s0 + d] < max_rain)
        lai = lai_curve(s0 + d + 0.5, tp_abs, g)
        etc = kc_rice(lai) * inp["et0"][s0 + d] * et_f
        x = np.arange(PER_DAY + 1) / PER_DAY - 0.5
        slope = (seg * x).sum(1) / (x ** 2).sum()
        rate = -slope + inp["rain"][s0 + d] - etc
        out_h[ok, d] = seg[ok].mean(1)
        out_tau[ok, d] = s0 + d + 0.5 - tp_abs[ok]
        out_r[ok, d] = rate[ok]
    return out_h, out_tau, out_r


def end_of_day_depth(h6):
    """Daily depth at the end of each day (the 00:00 reading of the next day), clipped to the pipe
    range; the last day has no next reading and is NaN. This is the state KGML-Paddy v1 estimates."""
    H = np.full((h6.shape[0], h6.shape[1] // PER_DAY), np.nan, np.float32)
    H[:, :-1] = np.clip(h6[:, PER_DAY::PER_DAY], PIPE_MIN, PIPE_MAX)
    return H


def world(name, **overrides):
    return replace(WORLDS[name], **overrides)
