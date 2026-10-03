"""Field-scale seepage + percolation (S&P) from satellite ponding sequences.

A ponded paddy loses water at L = ETc + SP (mm/d). An irrigation of depth h0
therefore stays visible as standing water for D = (h0 + rain) / L days. The
satellite never sees D directly: it sees wet/dry states at irregular clear-sky
dates. A wet run bounded by dry observations brackets D:

    dry(t0) ... wet(t1) ... wet(tk) ... dry(tk+1)
    irrigation time tau in (t0, t1]   ->   D in [tk - t1, tk+1 - t0]

With tau logged by the farmer (e.g. in the SmartSoil app) the bracket
tightens to [tk - tau, tk+1 - tau]. With pump hours logged, h0 is known too.

Given a lognormal prior on h0, each run gives a closed-form likelihood

    P(D in [a, b] | SP) = F_h0(b L - R) - F_h0(a L - R),   L = Kc ET0 + SP,

and the per-field posterior over SP is evaluated on a grid.
"""

import numpy as np
from scipy.stats import norm

SP_GRID = np.arange(0.1, 40.01, 0.1)       # mm/d
SP_PRIOR = norm.pdf(np.log(SP_GRID), np.log(5.0), 0.8) / SP_GRID  # lognormal, median 5 mm/d
H0_MEDIAN, H0_SIGMA = 50.0, 0.30           # irrigation depth prior (mm), Boro practice
EPS = 0.05                                  # robustness floor for misclassified runs
D_MAX = 40.0


def kc_rice(lai):
    """FAO-56 rice Kc under ponding: ~1.05 at transplanting rising to ~1.2."""
    return 1.05 + 0.15 * np.clip(lai / 3.0, 0, 1)


def episodes(days, state, tau=None):
    """Wet runs bracketed by dry observations.

    days  sorted observation days (float)
    state 1 = standing water, 0 = none; unusable observations already removed
    tau   optional sorted array of farmer-logged irrigation days
    Returns a list of dicts: a, b (duration bracket, days), w0, w1 (window).
    """
    out = []
    n = len(days)
    i = 1
    while i < n:
        if state[i] == 1 and state[i - 1] == 0:
            k = i
            while k + 1 < n and state[k + 1] == 1:
                k += 1
            if k + 1 < n:  # closed by a dry observation
                t0, t1, tk, tk1 = days[i - 1], days[i], days[k], days[k + 1]
                start_lo, start_hi = t0, t1
                if tau is not None:
                    logged = tau[(tau > t0) & (tau <= t1)]
                    if logged.size:
                        start_lo = start_hi = logged[-1]
                out.append({"a": max(tk - start_hi, 0.0), "b": tk1 - start_lo, "w0": start_lo, "w1": tk1})
            i = k + 1
        else:
            i += 1
    return out


def posterior(eps, et0_fn, rain_fn, kc=1.1, h0=None, h0_sigma=H0_SIGMA):
    """Posterior over SP_GRID for one field.

    et0_fn(w0, w1) mean ET0 over a window; rain_fn(w0, w1) total rain.
    kc: a constant, or kc(w0, w1) from the satellite LAI of that window.
    h0: optional list of logged irrigation depths (mm), one per episode.
    """
    logp = np.log(SP_PRIOR)
    for j, e in enumerate(eps):
        k = kc(e["w0"], e["w1"]) if callable(kc) else kc
        L = k * et0_fn(e["w0"], e["w1"]) + SP_GRID
        R = rain_fn(e["w0"], e["w1"])
        mu = np.log(h0[j]) if h0 is not None else np.log(H0_MEDIAN)
        sg = 0.1 if h0 is not None else h0_sigma
        hi = np.clip(e["b"] * L - R, 1e-6, None)
        lo = np.clip(e["a"] * L - R, 1e-6, None)
        p = norm.cdf((np.log(hi) - mu) / sg) - norm.cdf((np.log(lo) - mu) / sg)
        p = (1 - EPS) * p + EPS * min((e["b"] - e["a"]) / D_MAX, 1.0)
        logp += np.log(np.clip(p, 1e-300, None))
    post = np.exp(logp - logp.max())
    return post / post.sum()


def summarize(post):
    c = np.cumsum(post)
    q = lambda x: float(SP_GRID[np.searchsorted(c, x)])  # noqa: E731
    return {"median": q(0.5), "q05": q(0.05), "q95": q(0.95)}
