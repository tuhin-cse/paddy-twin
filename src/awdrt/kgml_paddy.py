"""KGML-Paddy v0: amortised posterior over each field's S&P law, with physics in the loss.

Structure (Layer 2):

    daily observations (satellite wet/dry, LAI, weather, farmer log)  ─┐
    static field attributes                                           ─┴─► encoder (BiGRU)
        ─► q(u | x) = N(mu, sigma), diagonal, in unconstrained space
        ─► theta = (a, b, c, w) = (e^u0, e^u1, c_max sig(u2), w_max sig(u3))
        ─► SP(h, tau) = a (1 + w tau/80) e^(-c tau) + b h            (the physics)

The positivity of a and b, S&P rising with depth (b > 0) and bounded sealing and
water-table terms are built into the parameterisation, so they hold for every output.

Training (the KGML-ag-Carbon recipe, adapted):
  * pre-train on simulated fields: Gaussian NLL of the true u (neural posterior estimation);
  * fine-tune on logger fields: the logger gives daily S&P on clean recession days
    (awdrt.paddy_sim.recession_rates); the loss is the expected Gaussian NLL of those rates
    under SP(h, tau; theta), with theta sampled from q (reparameterised), so loggers supervise
    the network through the physics. Simulated batches are replayed so the network does not
    forget the simulator, and the encoder runs at a lower learning rate than the head.
"""

import numpy as np
import torch
from torch import nn

from . import paddy_sim as ps

C_MAX, W_MAX = ps.BOUNDS["c_max"], ps.BOUNDS["w_max"]
N_PAR = 4
EPS = 1e-3


def theta_to_u(theta):
    a, b, c, w = (theta[:, i] for i in range(4))
    cl = np.clip(c / C_MAX, EPS, 1 - EPS)
    wl = np.clip(w / W_MAX, EPS, 1 - EPS)
    return np.stack([np.log(a), np.log(b), np.log(cl / (1 - cl)), np.log(wl / (1 - wl))], 1).astype(np.float32)


def u_to_theta(u):
    """Works on numpy arrays and torch tensors; last axis = 4 parameters."""
    if isinstance(u, torch.Tensor):
        return torch.stack([u[..., 0].exp(), u[..., 1].exp(), C_MAX * torch.sigmoid(u[..., 2]),
                            W_MAX * torch.sigmoid(u[..., 3])], -1)
    sig = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
    return np.stack([np.exp(u[..., 0]), np.exp(u[..., 1]), C_MAX * sig(u[..., 2]), W_MAX * sig(u[..., 3])], -1)


def sp_torch(theta, h, tau):
    """theta (..., 4); h, tau broadcastable to theta[..., 0]."""
    a, b, c, w = theta.unbind(-1)
    tau = tau.clamp(min=0)
    return a * (1 + w * tau / 80) * torch.exp(-c * tau) + b * h.clamp(min=0)


class Encoder(nn.Module):
    def __init__(self, n_ch=len(ps.CHANNELS), n_static=ps.N_STATIC, hidden=64):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_ch + n_static, hidden), nn.GELU())
        self.gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(4 * hidden + n_static, 128), nn.GELU(),
                                  nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 2 * N_PAR))

    def encode(self, X, static):
        """Per-day GRU outputs (B, T, 2 hidden)."""
        T = X.shape[1]
        z = self.inp(torch.cat([X, static[:, None, :].expand(-1, T, -1)], -1))
        return self.gru(z)[0]

    def forward(self, X, static):
        out = self.encode(X, static)
        m = X[:, :, :1]  # season mask
        mean = (out * m).sum(1) / m.sum(1).clamp(min=1)
        mx = out.masked_fill(m == 0, -1e4).max(1).values
        p = self.head(torch.cat([mean, mx, static], -1))
        mu, log_sd = p[:, :N_PAR], p[:, N_PAR:].clamp(-5, 3)
        return mu, log_sd


# ── losses ─────────────────────────────────────────────────────────────────────

def nll_u(mu, log_sd, u):
    return (0.5 * ((u - mu) / log_sd.exp()) ** 2 + log_sd).sum(-1).mean()


def recession_nll(mu, log_sd, h, tau, r, sigma=4.0, k=8):
    """Expected NLL of logger recession rates; h, tau, r (B, D) with NaN padding."""
    m = torch.isfinite(r)
    h, tau, r = (torch.nan_to_num(x) for x in (h, tau, r))
    eps = torch.randn(k, *mu.shape)
    theta = u_to_theta(mu + log_sd.exp() * eps)                 # (k, B, 4)
    pred = sp_torch(theta[:, :, None, :], h[None], tau[None])  # (k, B, D)
    ll = (0.5 * ((pred - r[None]) / sigma) ** 2) * m[None]
    return ll.sum(-1).mean() / m.sum(-1).float().clamp(min=1).mean()


# ── training ───────────────────────────────────────────────────────────────────

def to_t(*arrs):
    return [torch.as_tensor(np.asarray(a, np.float32)) for a in arrs]


def pretrain(sim, val, epochs=30, batch=256, lr=1e-3, seed=0, log=print):
    torch.manual_seed(seed)
    model = Encoder()
    X, S, U = to_t(sim["X"], sim["static"], theta_to_u(sim["theta"]))
    Xv, Sv, Uv = to_t(val["X"], val["static"], theta_to_u(val["theta"]))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state = np.inf, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), batch):
            j = perm[i:i + batch]
            loss = nll_u(*model(X[j], S[j]), U[j])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = float(nll_u(*model(Xv, Sv), Uv))
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
        log(f"  epoch {ep + 1:2d}  val NLL {vl:.3f}")
    model.load_state_dict(best_state)
    return model


def fine_tune(model, logger_set, replay, steps=300, lr_head=1e-3, lr_enc=2e-4, replay_w=1.0,
              replay_batch=128, sigma=4.0, seed=0):
    """logger_set: dict with X, static, h, tau, r (recession arrays). replay: a simulated set, or
    None for training from scratch on the loggers alone."""
    torch.manual_seed(seed)
    enc_params = list(model.inp.parameters()) + list(model.gru.parameters())
    opt = torch.optim.Adam([{"params": enc_params, "lr": lr_enc}, {"params": model.head.parameters(), "lr": lr_head}])
    X, S, h, tau, r = to_t(logger_set["X"], logger_set["static"], logger_set["h"], logger_set["tau"], logger_set["r"])
    if replay is not None:
        Xr, Sr, Ur = to_t(replay["X"], replay["static"], theta_to_u(replay["theta"]))
    model.train()
    for _ in range(steps):
        loss = recession_nll(*model(X, S), h, tau, r, sigma=sigma)
        if replay is not None:
            j = torch.randint(len(Xr), (replay_batch,))
            loss = loss + replay_w * nll_u(*model(Xr[j], Sr[j]), Ur[j])
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def posterior(model, X, static, n_samples=200, seed=0, batch=512):
    """Posterior samples of theta: (n_samples, n_fields, 4)."""
    model.eval()
    g = torch.Generator().manual_seed(seed)
    out = []
    Xt, St = to_t(X, static)
    for i in range(0, len(Xt), batch):
        mu, log_sd = model(Xt[i:i + batch], St[i:i + batch])
        eps = torch.randn(n_samples, *mu.shape, generator=g)
        out.append(u_to_theta(mu + log_sd.exp() * eps).numpy())
    return np.concatenate(out, 1)


def logger_theta(logger_set, c0=0.006, w0=0.25, min_days=8):
    """Each logger field's S&P law from its own recession days alone: (n, 4) theta, NaN rows where
    there are fewer than min_days clean days.

    Least squares of rate = a (1 + w0 tau/80) e^(-c0 tau) + b h; c and w are barely identifiable
    from one season, so they are fixed at mid-prior values.
    """
    h, tau, r = logger_set["h"], logger_set["tau"], logger_set["r"]
    out = np.full((len(r), 4), np.nan)
    for i in range(len(r)):
        m = np.isfinite(r[i])
        if m.sum() < min_days:
            continue
        A = np.c_[(1 + w0 * tau[i, m] / 80) * np.exp(-c0 * tau[i, m]), h[i, m]]
        (a, b), *_ = np.linalg.lstsq(A, r[i, m], rcond=None)
        out[i] = (max(a, 0.1), max(b, 0.0), c0, w0)
    return out


def logger_sp_eff(logger_set, inp, c0=0.006, w0=0.25, min_days=8):
    """Each logger field's effective S&P (50 mm on 1 March) from its own recession days alone.
    In simulator v2 this recovers the truth with about 18 % scatter (log sd 0.18)."""
    th = logger_theta(logger_set, c0, w0, min_days)
    out = np.full(len(th), np.nan)
    ok = np.isfinite(th[:, 0])
    if ok.any():
        out[ok] = ps.sp_effective(*th[ok].T, logger_set["tp_abs"][ok], inp, dt=0.05)
    return out


def sp_eff_samples(model, data, inp, n_samples=100, seed=0):
    th = posterior(model, data["X"], data["static"], n_samples, seed)
    return ps.sp_effective(*(th[..., i] for i in range(4)), data["tp_abs"][None, :], inp, dt=0.05)


def conformal_scale(sp_samples, sp_logger, target=0.9):
    """Split-conformal factor for the posterior's 90 % log-interval half-width.

    Scores are |log(logger estimate) - log(posterior median)| / half-width on logger fields the
    model was not trained on. The logger estimate carries its own error, so the factor is
    slightly conservative. Returns 1 if there are too few fields to calibrate.
    """
    med = np.median(sp_samples, 0)
    hw = (np.log(np.quantile(sp_samples, 0.95, 0)) - np.log(np.quantile(sp_samples, 0.05, 0))) / 2
    ok = np.isfinite(sp_logger) & (sp_logger > 0)
    n = ok.sum()
    if n < 5:
        return 1.0
    score = np.abs(np.log(sp_logger[ok]) - np.log(med[ok])) / hw[ok]
    level = min(1.0, np.ceil((n + 1) * target) / n)
    return float(np.quantile(score, level))


# ══ v1: daily water-level (state) head, physics consistency, causal forecasting ════════════════

CH = {c: i for i, c in enumerate(ps.CHANNELS)}
# channels that are observations (hidden after the forecast day); weather, season and crop age stay
OBS_CH = [CH[c] for c in ("lai_obs", "s2_wet", "s2_dry", "s1_wet", "s1_dry", "nisar_wet", "nisar_dry",
                          "irr_logged", "depth_logged", "gone")]
H_SCALE = 100.0  # mm per unit of the state head


def with_cut(X, cut):
    """Hide observations on days >= cut (per field) and append an 'observed' channel.

    X (B, T, C) tensor; cut (B,) long tensor of the first hidden day (T = nothing hidden).
    """
    T = X.shape[1]
    avail = (torch.arange(T)[None, :] < cut[:, None]).float()
    Xc = X.clone()
    Xc[:, :, OBS_CH] = Xc[:, :, OBS_CH] * avail[:, :, None]
    return torch.cat([Xc, avail[:, :, None]], -1)


class PaddyNet(Encoder):
    """Encoder + a daily state head: posterior over the S&P law and, for the end of each day, the
    probability that the field is ponded, the ponded depth (mm) and the depth below the surface (mm,
    negative, as a logger in an AWD pipe reads it). A single Gaussian cannot represent the depth,
    which is bimodal (ponded, or drained well below the surface); it hedges between the two."""

    def __init__(self, hidden=64, structure=True):
        super().__init__(n_ch=len(ps.CHANNELS) + 1, hidden=hidden)
        self.state = nn.Sequential(nn.Linear(2 * hidden, 64), nn.GELU(), nn.Linear(64, 5))
        # structure=False (ablation): no S&P law; the head predicts log effective S&P directly and it is
        # used as a constant loss rate, i.e. theta = (SP_eff, 0, 0, 0)
        self.structure = structure
        self.n_out = N_PAR if structure else 1
        if not structure:
            self.head[-1] = nn.Linear(64, 2)

    def forward(self, X, static):
        out = self.encode(X, static)
        m = X[:, :, :1]
        mean = (out * m).sum(1) / m.sum(1).clamp(min=1)
        mx = out.masked_fill(m == 0, -1e4).max(1).values
        p = self.head(torch.cat([mean, mx, static], -1))
        st = self.state(out)
        state = {"logit_wet": st[..., 0],
                 "pos_mu": nn.functional.softplus(st[..., 1]) * H_SCALE, "pos_lsd": st[..., 2].clamp(-4, 3),
                 "neg_mu": -nn.functional.softplus(st[..., 3]) * H_SCALE, "neg_lsd": st[..., 4].clamp(-4, 3)}
        return p[:, :self.n_out], p[:, self.n_out:].clamp(-5, 3), state


def model_theta(model, u):
    """Posterior sample(s) in the model's output space -> S&P law theta (..., 4). Without structure the
    single output is log effective S&P, used as a constant loss rate."""
    if getattr(model, "structure", True):
        return u_to_theta(u)
    a = u[..., 0].exp() if isinstance(u, torch.Tensor) else np.exp(u[..., 0])
    z = torch.zeros_like(a) if isinstance(a, torch.Tensor) else np.zeros_like(a)
    stack = torch.stack if isinstance(a, torch.Tensor) else np.stack
    return stack([a, z, z, z], -1)


SAT_CH = [CH[c] for c in ("lai_obs", "s2_wet", "s2_dry", "s1_wet", "s1_dry", "nisar_wet", "nisar_dry")]


def drop_satellite(X):
    """Remove every satellite input (wet/dry passes and satellite LAI) - ablation."""
    X = X.clone() if isinstance(X, torch.Tensor) else X.copy()
    X[:, :, SAT_CH] = 0
    return X


def state_loss(st, H, mask):
    """Ponded or not (cross-entropy) + depth NLL within each regime; H (B, T) mm; mask where it counts."""
    wet = (H > 0).float()
    bce = nn.functional.binary_cross_entropy_with_logits(st["logit_wet"], wet, reduction="none")

    def nll(mu, lsd):
        z = (H - mu) / H_SCALE
        return 0.5 * (z / lsd.exp()) ** 2 + lsd

    ll = bce + wet * nll(st["pos_mu"], st["pos_lsd"]) + (1 - wet) * nll(st["neg_mu"], st["neg_lsd"])
    return (ll * mask).sum() / mask.sum().clamp(min=1)


def physics_residual(theta, h, X, mask=None, min_h=5.0, w_up=0.05):
    """Water-balance consistency of a daily ponded-depth path with the field's S&P law (mm/d).

    On days that start and end ponded (mask, or h > min_h then h > 0), the modelled change must
    match rain - Kc ET0 - SP(h_t, tau). Drops faster than the physics are penalised fully; rises
    lightly (w_up), since an irrigation the farmer did not log looks exactly like that.
    """
    h0, h1 = h[:, :-1], h[:, 1:]
    et0 = X[:, :-1, CH["et0"]] * 10
    rain = X[:, :-1, CH["rain"]] * 20
    lai = X[:, :-1, CH["lai_obs"]] * 5
    kc = 1.05 + 0.15 * (lai / 3).clamp(0, 1)
    tau = X[:, :-1, CH["days_since_tp"]] * 100 + 0.5
    sp = sp_torch(theta[:, None, :], h0, tau)
    r = h1 - (h0 + rain - kc * et0 - sp)
    m = ((h0 > min_h) & (h1 > 0)).float() if mask is None else mask[:, :-1] * mask[:, 1:]
    m = m * (X[:, :-1, CH["season"]] > 0).float()
    loss = (torch.relu(-r) ** 2 + w_up * torch.relu(r) ** 2) * m
    return loss.sum() / m.sum().clamp(min=1)


LOG_CH = [CH[c] for c in ("irr_logged", "depth_logged", "gone")]


def drop_logs(X, mask):
    """Remove the farmer's log (irrigation taps, pump hours, "water gone") for fields where mask is true."""
    X = X.clone() if isinstance(X, torch.Tensor) else X.copy()
    idx = np.flatnonzero(np.asarray(mask))
    for c in LOG_CH:
        X[idx, :, c] = 0
    return X


def pretrain_v1(sim, val, epochs=30, batch=256, lr=1e-3, p_cut=0.5, min_cut=30, phys_w=0.01, seed=0, log=print,
                p_nolog=0.0, structure=True, sat_off=False):
    """Pre-train PaddyNet on simulated fields: parameter NLL + daily state loss (up to the cut) +
    physics consistency on days the network itself judges ponded. With probability p_cut a field's
    observations are hidden from a random day on, so the same network serves for season-end
    estimation and for forecasting from today. With probability p_nolog a field's farmer log is removed
    (a farmer who does not use the app), in each batch afresh; validation uses a fixed share of such
    fields. p_nolog = 0 reproduces v1 exactly (its random streams are untouched).
    Ablation switches: structure=False (no S&P law; see PaddyNet), sat_off=True (no satellite inputs),
    phys_w=0 (no physics-consistency loss)."""
    torch.manual_seed(seed)
    model = PaddyNet(structure=structure)
    target = (lambda d: theta_to_u(d["theta"])) if structure else (lambda d: np.log(d["sp_eff"])[:, None].astype(np.float32))
    X, S, U, H = to_t(drop_satellite(sim["X"]) if sat_off else sim["X"], sim["static"], target(sim),
                      np.nan_to_num(sim["H"], nan=0.0))
    M = torch.as_tensor(np.isfinite(sim["H"]) & (sim["X"][:, :, CH["season"]] > 0), dtype=torch.float32)
    Xv, Sv, Uv, Hv = to_t(drop_satellite(val["X"]) if sat_off else val["X"], val["static"], target(val),
                          np.nan_to_num(val["H"], nan=0.0))
    Mv = torch.as_tensor(np.isfinite(val["H"]) & (val["X"][:, :, CH["season"]] > 0), dtype=torch.float32)
    T = X.shape[1]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    g = torch.Generator().manual_seed(seed)
    g_nolog = np.random.default_rng(seed + 1)
    if p_nolog > 0:
        Xv = drop_logs(Xv, np.random.default_rng(seed + 2).random(len(Xv)) < p_nolog)
    best, best_state = np.inf, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), batch):
            j = perm[i:i + batch]
            cut = torch.where(torch.rand(len(j), generator=g) < p_cut,
                              torch.randint(min_cut, T, (len(j),), generator=g), torch.full((len(j),), T))
            Xj = drop_logs(X[j], g_nolog.random(len(j)) < p_nolog) if p_nolog > 0 else X[j]
            mu, lsd, st = model(with_cut(Xj, cut), S[j])
            # the state is only asked for up to the day before the cut (the forecast starts from there)
            mj = M[j] * (torch.arange(T)[None, :] < cut[:, None]).float()
            ponded = (torch.sigmoid(st["logit_wet"]) > 0.5).float().detach()
            loss = nll_u(mu, lsd, U[j]) + state_loss(st, H[j], mj) + \
                phys_w * physics_residual(model_theta(model, mu), st["pos_mu"], Xj, mask=ponded)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            mu, lsd, st = model(with_cut(Xv, torch.full((len(Xv),), T)), Sv)
            vl = float(nll_u(mu, lsd, Uv) + state_loss(st, Hv, Mv))
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
        log(f"  epoch {ep + 1:2d}  val NLL {vl:.3f}")
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_v1(model, X, static, cut, n_samples=100, seed=0, batch=512, progress=None):
    """Posterior samples of theta (k, n, 4) and the daily state (dict of (n, T) arrays: p_wet, pos_mu,
    pos_sd, neg_mu, neg_sd, and the point estimate h), using observations before day `cut` only
    (int or per-field array). progress(fraction) is called after each batch, if given."""
    model.eval()
    g = torch.Generator().manual_seed(seed)
    Xt, St = to_t(X, static)
    cut_t = torch.as_tensor(np.broadcast_to(cut, (len(Xt),)).copy(), dtype=torch.long)
    th, parts = [], {k: [] for k in ("p_wet", "pos_mu", "pos_sd", "neg_mu", "neg_sd")}
    for i in range(0, len(Xt), batch):
        mu, lsd, st = model(with_cut(Xt[i:i + batch], cut_t[i:i + batch]), St[i:i + batch])
        eps = torch.randn(n_samples, *mu.shape, generator=g)
        th.append(model_theta(model, mu + lsd.exp() * eps).numpy())
        parts["p_wet"].append(torch.sigmoid(st["logit_wet"]).numpy())
        for k in ("pos", "neg"):
            parts[f"{k}_mu"].append(st[f"{k}_mu"].numpy())
            parts[f"{k}_sd"].append((st[f"{k}_lsd"].exp() * H_SCALE).numpy())
        if progress:
            progress(min(1.0, (i + batch) / len(Xt)))
    state = {k: np.concatenate(v) for k, v in parts.items()}
    state["h"] = np.where(state["p_wet"] > 0.5, state["pos_mu"], state["neg_mu"])
    return np.concatenate(th, 1), state


def sample_depth(state, idx_day, n_samples, rng, fields=slice(None)):
    """Depth samples (k, n) at the end of day idx_day from the mixture: ponded with probability
    p_wet (ponded depth ~ N(pos_mu, pos_sd), floored at 0.1 mm), otherwise the below-surface depth."""
    p = state["p_wet"][fields, idx_day]
    wet = rng.random((n_samples, len(p))) < p[None, :]
    pos = np.maximum(state["pos_mu"][fields, idx_day] + state["pos_sd"][fields, idx_day] * rng.standard_normal(wet.shape), 0.1)
    neg = np.minimum(state["neg_mu"][fields, idx_day] + state["neg_sd"][fields, idx_day] * rng.standard_normal(wet.shape), 0.0)
    return np.where(wet, pos, neg)


def forecast_dry(theta, h0, t0_abs, tp_abs, inp, et_f=1.0, g=1.0, max_days=30.0, dt=0.05, progress=None):
    """Days until the ponding ends, starting from depth h0 (mm) at absolute day t0_abs, with no
    irrigation and the ERA5 weather that followed. Arrays broadcast (samples (k, n) or truths (n,)).
    Fields already dry (h0 <= 0) return 0; fields still ponded after max_days return max_days.
    progress(fraction) is called every 10 steps, if given."""
    a, b, c, w, h, tp, t0, ef, gg = np.broadcast_arrays(*(np.asarray(x, float) for x in (
        theta[..., 0], theta[..., 1], theta[..., 2], theta[..., 3], h0, tp_abs, t0_abs, et_f, g)))
    h = h.copy()
    D = np.where(h <= 0, 0.0, max_days)
    live = h > 0
    for k in range(int(max_days / dt)):
        t = t0 + k * dt
        d = t.astype(int)
        etc = ps.kc_rice(ps.lai_curve(t, tp, gg)) * inp["et0"][d] * ef
        h = h + live * (inp["rain"][d] - etc - ps.sp_rate(a, b, c, w, h, t - tp)) * dt
        newly = live & (h <= 0)
        D = np.where(newly, (k + 1) * dt, D)
        live &= ~newly
        if progress and k % 10 == 0:
            progress(k * dt / max_days)
        if not live.any():
            break
    return D


def fine_tune_v1(model, logger_set, replay, steps=300, lr_head=1e-3, lr_enc=2e-4, p_cut=0.5, min_cut=30,
                 replay_w=1.0, replay_batch=128, rec_w=1.0, phys_w=0.01, seed=0, anchor_w=0.0, val_fn=None,
                 eval_every=25):
    """Fine-tune PaddyNet on logger fields.

    logger_set: X, static, H_log (end-of-day logger level, mm) and the recession arrays h, tau, r
    (without structure, sp_log instead: each field's effective S&P from its own logger).
    Loss on the loggers: the daily state loss against the logger level (observations hidden after a
    random day, as in pre-training) + the recession-rate NLL of the S&P law (as v0) + physics
    consistency. Simulated fields are replayed (parameter NLL + state loss) so the network keeps what
    it learned from the simulator; the encoder runs at a lower learning rate than the heads.

    Regularisation (Experiment 21): anchor_w > 0 adds an L2-SP penalty, anchor_w * sum (p - p0)^2 over
    the weights p0 the fine-tuning starts from. val_fn(model) -> score (lower is better), if given, is
    evaluated every eval_every steps (and at step 0); the best weights are kept (early stopping).
    Learning rates of 0 freeze that part.
    """
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    enc = list(model.inp.parameters()) + list(model.gru.parameters())
    opt = torch.optim.Adam([{"params": enc, "lr": lr_enc},
                            {"params": list(model.head.parameters()) + list(model.state.parameters()), "lr": lr_head}])
    X, S, H = to_t(logger_set["X"], logger_set["static"], np.nan_to_num(logger_set["H_log"], nan=0.0))
    if model.structure:
        h, tau, r = to_t(logger_set["h"], logger_set["tau"], logger_set["r"])
    else:  # no S&P law (ablation): Gaussian NLL of log effective S&P, on the loggers and the replay
        sp = np.asarray(logger_set["sp_log"], np.float32)
        ok = torch.as_tensor(np.isfinite(sp) & (sp > 0), dtype=torch.float32)
        y = torch.as_tensor(np.log(np.where(np.isfinite(sp) & (sp > 0), sp, 1.0)), dtype=torch.float32)
    M = torch.as_tensor(np.isfinite(logger_set["H_log"]) & (logger_set["X"][:, :, CH["season"]] > 0), dtype=torch.float32)
    Ur = theta_to_u(replay["theta"]) if model.structure else np.log(replay["sp_eff"])[:, None].astype(np.float32)
    Xr, Sr, Ur, Hr = to_t(replay["X"], replay["static"], Ur, np.nan_to_num(replay["H"], nan=0.0))
    Mr = torch.as_tensor(np.isfinite(replay["H"]) & (replay["X"][:, :, CH["season"]] > 0), dtype=torch.float32)
    T = X.shape[1]

    def cuts(n):
        return torch.where(torch.rand(n, generator=g) < p_cut, torch.randint(min_cut, T, (n,), generator=g),
                           torch.full((n,), T))

    p0 = [q.detach().clone() for q in model.parameters()]
    best = (val_fn(model), {k: v.clone() for k, v in model.state_dict().items()}) if val_fn else None
    model.train()
    for step in range(steps):
        c = cuts(len(X))
        mu, lsd, st = model(with_cut(X, c), S)
        mask = M * (torch.arange(T)[None, :] < c[:, None]).float()
        ponded = (torch.sigmoid(st["logit_wet"]) > 0.5).float().detach()
        mu_full, lsd_full, _ = model(with_cut(X, torch.full((len(X),), T)), S)
        if model.structure:
            target_loss = recession_nll(mu_full, lsd_full, h, tau, r)
        else:
            z = (y - mu_full[:, 0]) / lsd_full[:, 0].exp()
            target_loss = ((0.5 * z ** 2 + lsd_full[:, 0]) * ok).sum() / ok.sum().clamp(min=1)
        loss = state_loss(st, H, mask) + rec_w * target_loss + \
            phys_w * physics_residual(model_theta(model, mu), st["pos_mu"], X, mask=ponded)
        j = torch.randint(len(Xr), (replay_batch,), generator=g)
        cr = cuts(replay_batch)
        mu_r, lsd_r, st_r = model(with_cut(Xr[j], cr), Sr[j])
        mask_r = Mr[j] * (torch.arange(T)[None, :] < cr[:, None]).float()
        loss = loss + replay_w * (nll_u(mu_r, lsd_r, Ur[j]) + state_loss(st_r, Hr[j], mask_r))
        if anchor_w > 0:
            loss = loss + anchor_w * sum(((q - q0) ** 2).sum() for q, q0 in zip(model.parameters(), p0))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if val_fn and (step + 1) % eval_every == 0:
            score = val_fn(model)
            if score < best[0]:
                best = (score, {k: v.clone() for k, v in model.state_dict().items()})
            model.train()
    if val_fn:
        model.load_state_dict(best[1])
        model.best_step_score = best[0]
    model.eval()
    return model


# ══ farmer answers: "is there water in the field now?" ═══════════════════════════════════════════

DEPTH_CATS = (0.0, 20.0, 50.0)  # "a little" (0-20 mm), "some" (20-50), "a lot" (> 50); 0 = no water


def depth_category(h):
    """0 = no standing water, 1 = a little (<= 20 mm), 2 = some (<= 50 mm), 3 = a lot."""
    h = np.asarray(h)
    return np.where(h <= 0, 0, np.where(h <= DEPTH_CATS[1], 1, np.where(h <= DEPTH_CATS[2], 2, 3)))


def answer_likelihood(h_samples, answer, kind, p_correct=0.95, cat_correct=0.8, gauge_sd=5.0):
    """Likelihood of a farmer's answer under each depth sample (k, n) -> (k, n).

    kind "yesno":  answer 1/0 for water / no water; the farmer is right with probability p_correct.
    kind "depth":  answer a category 0-3 (depth_category); "no water" vs "water" is right with
                   p_correct, and among water answers the category is right with cat_correct, else one
                   step off (split between neighbours).
    kind "gauge":  answer a reading in mm from a staff gauge, Gaussian with gauge_sd; a reading of 0 or
                   below means no standing water.
    """
    h = np.asarray(h_samples)
    a = np.asarray(answer)[None, :]
    if kind == "yesno":
        wet = h > 0
        return np.where(wet == (a > 0), p_correct, 1 - p_correct)
    if kind == "depth":
        c = depth_category(h)
        wet_ok = np.where((c > 0) == (a > 0), p_correct, 1 - p_correct)
        n_nb = np.where((c == 1) | (c == 3), 1, 2)  # categories 1 and 3 have one neighbour, 2 has two
        within = np.where(c == a, cat_correct, np.where(np.abs(c - a) == 1, (1 - cat_correct) / n_nb, 1e-3))
        return wet_ok * np.where((c > 0) & (a > 0), within, 1.0)
    if kind == "gauge":
        return np.where(a > 0, np.exp(-0.5 * ((h - a) / gauge_sd) ** 2) * (h > 0) + 1e-6,
                        np.where(h <= 0, 1.0, np.exp(-0.5 * (h / gauge_sd) ** 2)))
    raise ValueError(kind)


def weighted_quantiles(values, weights, qs):
    """Per-column weighted quantiles: values, weights (k, n) -> (len(qs), n)."""
    o = np.argsort(values, 0)
    v = np.take_along_axis(values, o, 0)
    w = np.take_along_axis(weights, o, 0)
    cw = np.cumsum(w, 0)
    cw = cw / cw[-1:]
    return np.stack([v[np.argmax(cw >= q, 0), np.arange(v.shape[1])] for q in qs])


# ══ N3: combining a satellite pass with the physics-informed prior ═══════════════════════════════

def pass_rates(inp, sensor, lai, rec_shift=0.0, spec_shift=0.0):
    """Recall and specificity of a wet/dry pass at the observed LAI. With no shifts these are the
    nominal rates (as the simulator's prior world uses them): Sentinel-2 from Experiment 1, Sentinel-1
    from the measured accuracy (03b). Shifts give rates re-estimated against loggers (calibrated)."""
    lai = np.asarray(lai, float)
    if sensor == "s2":
        lo, rec, spec = inp["lai_lo"], inp["rec"] + rec_shift, inp["spec"] + spec_shift
    elif sensor == "s1":
        lo = np.array([b[0] for b in inp["s1_bins"]])
        rec = np.array([b[1] for b in inp["s1_bins"]]) + rec_shift
        spec = np.array([b[1] for b in inp["s1_bins"]]) + spec_shift
    else:
        raise ValueError(sensor)
    b = np.clip(np.searchsorted(lo, lai, side="right") - 1, 0, len(lo) - 1)
    return np.clip(rec[b], 0.5, 1), np.clip(spec[b], 0.5, 1)


def fuse_pass(p_prior, said_wet, rec, spec):
    """Posterior P(ponded) after a wet/dry pass with the given recall and specificity (Bayes' rule)."""
    p = np.clip(p_prior, 1e-4, 1 - 1e-4)
    like_wet = np.where(said_wet, rec, 1 - rec)
    like_dry = np.where(said_wet, 1 - spec, spec)
    return p * like_wet / (p * like_wet + (1 - p) * like_dry)


def train_loggers_only(model, logger_set, steps=600, lr=1e-3, p_cut=0.5, min_cut=30, phys_w=0.01, seed=0):
    """Ablation "no pre-training": train a fresh PaddyNet on logger fields alone.

    logger_set: X, static, H_log, and either the recession arrays h, tau, r (structure) or sp_log, each
    field's effective S&P estimated from its own logger (no structure). Loss: state loss against the
    logger level (observations hidden after random days) + the S&P target + physics consistency."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X, S, H = to_t(logger_set["X"], logger_set["static"], np.nan_to_num(logger_set["H_log"], nan=0.0))
    M = torch.as_tensor(np.isfinite(logger_set["H_log"]) & (logger_set["X"][:, :, CH["season"]] > 0), dtype=torch.float32)
    if model.structure:
        h, tau, r = to_t(logger_set["h"], logger_set["tau"], logger_set["r"])
    else:
        sp = np.asarray(logger_set["sp_log"], np.float32)
        ok = torch.as_tensor(np.isfinite(sp) & (sp > 0), dtype=torch.float32)
        y = torch.as_tensor(np.log(np.where(np.isfinite(sp) & (sp > 0), sp, 1.0)), dtype=torch.float32)
    T = X.shape[1]
    model.train()
    for _ in range(steps):
        c = torch.where(torch.rand(len(X), generator=g) < p_cut, torch.randint(min_cut, T, (len(X),), generator=g),
                        torch.full((len(X),), T))
        mu, lsd, st = model(with_cut(X, c), S)
        mask = M * (torch.arange(T)[None, :] < c[:, None]).float()
        ponded = (torch.sigmoid(st["logit_wet"]) > 0.5).float().detach()
        mu_f, lsd_f, _ = model(with_cut(X, torch.full((len(X),), T)), S)
        if model.structure:
            target_loss = recession_nll(mu_f, lsd_f, h, tau, r)
        else:
            z = (y - mu_f[:, 0]) / lsd_f[:, 0].exp()
            target_loss = ((0.5 * z ** 2 + lsd_f[:, 0]) * ok).sum() / ok.sum().clamp(min=1)
        loss = state_loss(st, H, mask) + target_loss + \
            phys_w * physics_residual(model_theta(model, mu), st["pos_mu"], X, mask=ponded)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    return model
