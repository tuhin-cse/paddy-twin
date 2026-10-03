"""Arai-style baseline: per-field particle filter with a regional (spatially constant) loss law.

After Arai et al. (2022, Remote Sens. Environ. 279:113139): a paddy water-level model whose
water-loss parameters are fitted once (on ground loggers) and held constant over the region, and a
particle filter that assimilates satellite submergence observations to estimate, per pixel, the
water level and the farmer's irrigation threshold ("D before irrigation"). Arai et al. used
PALSAR-2; here the filter assimilates the same Sentinel-2 / Sentinel-1 passes as KGML-Paddy, with
their nominal error rates, so the comparison isolates the model, not the data.

Differences from the original, all in the baseline's favour or neutral: the loss law has the same
functional form as the simulator (awdrt.paddy_sim.sp_rate) with regional a, b; weather and crop
calendar are known; optionally the farmer's logged irrigations are applied ("+ logs" variant),
which Arai et al. did not have per pixel.

Particles per field: water level h (mm; below the surface via a regional specific yield), the
farmer's threshold D (prior uniform -150 .. 20 mm), a pending irrigation time and its depth, and (with
logs) the unknown factor between the logged depth and the water that arrives (pump rating). D and the
rating are static parameters, jittered slightly at each resampling; the level carries small process
noise, so the particle cloud does not collapse onto one trajectory.
"""

import numpy as np

from . import kgml_paddy as kp
from . import paddy_sim as ps

CH = kp.CH
DT = 0.25
SY_REG = 0.065           # regional specific yield (middle of the simulator's range)
D_PRIOR = (-150.0, 20.0)  # farmer threshold prior (mm)
H0_MED, H0_SIG = 55.0, 0.25
P_LOGGED = 0.8            # "+ logs" variant: share of the farmer's irrigations assumed to be in the log
RATING_SIG = 0.25         # "+ logs": prior sd (log) of the unknown pump-rating factor on logged depths
H_NOISE = 2.0             # process noise on the ponded level, mm per sqrt(day), against particle collapse


def regional_theta(logger_set, c0=0.006, w0=0.25):
    """Regional loss law from logger fields: medians of the per-logger least-squares a and b."""
    th = kp.logger_theta(logger_set, c0, w0)
    ok = np.isfinite(th[:, 0])
    return np.array([np.median(th[ok, 0]), np.median(th[ok, 1]), c0, w0])


def _resample(w, rng):
    """Systematic resampling per row: w (n, P) normalised -> indices (n, P)."""
    n, P = w.shape
    cw = np.cumsum(w, 1)
    cw[:, -1] = 1.0
    pos = (np.arange(P)[None, :] + rng.random((n, 1))) / P
    flat = (cw + np.arange(n)[:, None]).ravel()
    idx = np.searchsorted(flat, (pos + np.arange(n)[:, None]).ravel()).reshape(n, P) - np.arange(n)[:, None] * P
    return np.clip(idx, 0, P - 1)


def run(data, inp, theta_reg, snap_days, use_logs=False, P=200, seed=0):
    """Filter every field of `data` through the season; snapshot the particles at the start of each day
    in snap_days (before that day's observations). Returns {day: {"h": (n, P), "D": (n, P)}} and the
    season-end threshold particles.

    use_logs: logged irrigations are applied as known events. A particle whose level falls below its
    threshold then decides once whether that irrigation is one the farmer logs (probability P_LOGGED):
    if so it waits for the log instead of adding water itself, so logged irrigations are not counted
    twice; otherwise it irrigates on its own (an unlogged irrigation)."""
    rng = np.random.default_rng(seed)
    X = data["X"]
    n, T = X.shape[0], X.shape[1]
    s0 = inp["s0"]
    tp = data["tp_abs"][:, None]
    a, b, c, w = theta_reg
    h = np.full((n, P), np.nan)
    D = rng.uniform(*D_PRIOR, (n, P))
    pending = np.full((n, P), np.inf)
    depth = np.zeros((n, P))
    waiting = np.zeros((n, P), bool)  # "+ logs": threshold crossed, irrigation expected in the log
    rating = rng.lognormal(0.0, RATING_SIG, (n, P))
    started = np.zeros(n, bool)
    snaps = {}
    steps_per_day = int(round(1 / DT))
    logged = X[:, :, CH["irr_logged"]] > 0
    logged_depth = X[:, :, CH["depth_logged"]] * 100
    for day in range(T):
        if day in snap_days:
            snaps[day] = {"h": np.where(started[:, None], h, np.nan).copy(), "D": D.copy()}
        for k in range(steps_per_day):
            t = s0 + day + k * DT
            start = ~started & (tp[:, 0] <= t)
            if start.any():
                h[start] = rng.lognormal(np.log(H0_MED), H0_SIG, (start.sum(), P))
                started |= start
            live = started[:, None]
            # logged irrigation (the "+ logs" variant), applied at midday of the logged day
            if use_logs and k == 2:
                lg = logged[:, day] & started
                if lg.any():
                    h[lg] = _add_water(h[lg], logged_depth[lg, day][:, None] / rating[lg])
                    pending[lg] = np.inf
                    waiting[lg] = False
            fire = live & (t >= pending)
            if fire.any():
                h = np.where(fire, _add_water(h, depth), h)
                pending = np.where(fire, np.inf, pending)
            lai = ps.lai_curve(t, tp)
            etc = ps.kc_rice(lai) * inp["et0"][int(t)]
            tau = np.clip(t - tp, 0, None)
            base = a * (1 + w * tau / 80) * np.exp(-c * tau)
            pond = h > 0
            loss = inp["rain"][int(t)] - etc - np.where(pond, base + b * np.clip(h, 0, None), base)
            S = np.where(pond, h, h * SY_REG) + loss * DT
            S = np.maximum(S, ps.PIPE_MIN * SY_REG)
            h = np.where(live, np.where(S > 0, S, S / SY_REG), h)
            h = np.where(live & (h > 0), np.maximum(h + H_NOISE * np.sqrt(DT) * rng.standard_normal((n, P)), 0.0), h)
            waiting &= h < D  # the level recovered (rain, or a logged irrigation): stop waiting
            trig = live & np.isinf(pending) & ~waiting & (h < D)
            if use_logs and trig.any():
                to_log = trig & (rng.random((n, P)) < P_LOGGED)
                waiting |= to_log
                trig &= ~to_log
            if trig.any():
                pending = np.where(trig, t + rng.uniform(0.5, 2.0, (n, P)), pending)
                depth = np.where(trig, rng.lognormal(np.log(H0_MED), H0_SIG, (n, P)), depth)
            # assimilate passes: S2 at 12:00 (k == 2), S1 at 18:00 (k == 3)
            for sensor, kk in (("s2", 2), ("s1", 3)):
                if k != kk:
                    continue
                wet_o = X[:, day, CH[f"{sensor}_wet"]] > 0
                dry_o = X[:, day, CH[f"{sensor}_dry"]] > 0
                f = np.flatnonzero((wet_o | dry_o) & started)
                if not f.size:
                    continue
                rec, spec = kp.pass_rates(inp, sensor, X[f, day, CH["lai_obs"]] * 5)
                wet_p = h[f] > 0
                said = wet_o[f][:, None]
                like = np.where(wet_p, np.where(said, rec[:, None], 1 - rec[:, None]),
                                np.where(said, 1 - spec[:, None], spec[:, None]))
                wts = like / like.sum(1, keepdims=True)
                idx = _resample(wts, rng)
                h[f] = np.take_along_axis(h[f], idx, 1)
                D[f] = np.take_along_axis(D[f], idx, 1)
                pending[f] = np.take_along_axis(pending[f], idx, 1)
                waiting[f] = np.take_along_axis(waiting[f], idx, 1)
                rating[f] = np.take_along_axis(rating[f], idx, 1) * np.exp(rng.normal(0, 0.02, idx.shape))
                depth[f] = np.take_along_axis(depth[f], idx, 1)
                # jitter the static parameter a little so it does not collapse
                D[f] = np.clip(D[f] + rng.normal(0, 3.0, D[f].shape), *D_PRIOR)
    return snaps, D


def _add_water(h, add):
    """Add water (mm) to a level that may be below the surface (storage with regional Sy)."""
    S = np.where(h > 0, h, h * SY_REG) + add
    return np.where(S > 0, S, S / SY_REG)
