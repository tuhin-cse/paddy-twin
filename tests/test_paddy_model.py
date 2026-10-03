"""Fast checks of simulator v2 and KGML-Paddy v0 (python -m pytest tests)."""

import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from awdrt import kgml_paddy as kp  # noqa: E402
from awdrt import paddy_sim as ps  # noqa: E402

INP = ps.load_inputs()


def test_sp_effective_equals_constant_rate():
    a = np.array([2.0, 5.0, 12.0])
    got = ps.sp_effective(a, 0 * a, 0 * a, 0 * a, np.full(3, INP["s0"] + 5.0), INP, dt=0.01)
    assert np.allclose(got, a, atol=0.1)


def test_sp_effective_rises_with_head_term():
    a = np.full(2, 5.0)
    got = ps.sp_effective(a, np.array([0.0, 0.05]), 0 * a, 0 * a, np.full(2, INP["s0"] + 5.0), INP)
    assert got[1] > got[0] + 0.5


def test_simulator_shapes_and_realism():
    s = ps.simulate(ps.WORLDS["prior"], 200, 0, INP)
    T = INP["s_end"] - INP["s0"]
    assert s["X"].shape == (200, T, len(ps.CHANNELS))
    assert s["logger"].shape == (200, T * ps.PER_DAY)
    assert np.isfinite(s["sp_eff"]).all() and (s["sp_eff"] > 0).all()
    assert 3 <= np.median(s["n_irr"]) <= 30  # Boro: a few irrigations a month


def test_recession_rates_unbiased_without_noise_and_lateral():
    s = ps.simulate(ps.world("prior", k_lat_max=0.0, logger_noise=0.0), 200, 1, INP)
    h, tau, r = ps.recession_rates(s["logger"], s["tp_abs"], INP, et_f=s["et_f"], g=s["g"])
    a, b, c, w = (s["theta"][:, i:i + 1] for i in range(4))
    m = np.isfinite(r)
    assert m.sum() > 1000
    assert abs(np.mean(r[m] - ps.sp_rate(a, b, c, w, h, tau)[m])) < 0.05


def test_theta_transform_round_trip_and_torch_physics():
    s = ps.simulate(ps.WORLDS["reality"], 100, 2, INP)
    u = kp.theta_to_u(s["theta"])
    back = kp.u_to_theta(u)
    inside = (s["theta"][:, 2] > 1e-4) & (s["theta"][:, 3] > 1e-3)
    assert np.allclose(back[inside], s["theta"][inside], rtol=1e-3, atol=1e-5)
    h, tau = np.full(100, 30.0), np.full(100, 25.0)
    ref = ps.sp_rate(*back.T, h, tau)
    got = kp.sp_torch(torch.as_tensor(back), torch.as_tensor(h), torch.as_tensor(tau)).numpy()
    assert np.allclose(ref, got, rtol=1e-5)


def test_encoder_runs_and_losses_are_finite():
    s = ps.simulate(ps.WORLDS["prior"], 16, 3, INP)
    model = kp.Encoder()
    X, S, U = kp.to_t(s["X"], s["static"], kp.theta_to_u(s["theta"]))
    mu, log_sd = model(X, S)
    assert mu.shape == (16, kp.N_PAR)
    assert torch.isfinite(kp.nll_u(mu, log_sd, U))
    r = torch.full((16, 5), float("nan"))
    r[:, :2] = 5.0
    loss = kp.recession_nll(mu, log_sd, torch.full((16, 5), 30.0), torch.full((16, 5), 20.0), r)
    assert torch.isfinite(loss)


def test_forecast_dry_constant_rate_matches_hand_calculation():
    t0 = INP["t_mid"] + 0.0
    tp = t0 - 30.0
    th = np.array([[6.0, 0.0, 0.0, 0.0]])
    D = kp.forecast_dry(th, np.array([50.0]), t0, tp, INP, dt=0.01)
    # hand: integrate daily loss = Kc(LAI) ET0 + 6 - rain until 50 mm are gone
    h, t = 50.0, t0
    while h > 0:
        d = int(t)
        h -= (ps.kc_rice(ps.lai_curve(t, tp)) * INP["et0"][d] + 6.0 - INP["rain"][d]) * 0.01
        t += 0.01
    assert abs(D[0] - (t - t0)) < 0.05
    assert kp.forecast_dry(th, np.array([-5.0]), t0, tp, INP)[0] == 0.0


def test_physics_residual_zero_on_a_physical_path():
    s = ps.simulate(ps.WORLDS["prior"], 4, 5, INP)
    X = torch.as_tensor(s["X"][:1])
    theta = torch.tensor([[5.0, 0.03, 0.0, 0.0]])
    T = X.shape[1]
    h = torch.zeros(1, T)
    h[0, 0] = 80.0
    for t in range(T - 1):  # build the path with the same discretisation as the loss
        x = X[0, t]
        kc = 1.05 + 0.15 * (x[kp.CH["lai_obs"]] * 5 / 3).clamp(0, 1)
        tau = x[kp.CH["days_since_tp"]] * 100 + 0.5
        sp = kp.sp_torch(theta[0], h[0, t], tau)
        h[0, t + 1] = h[0, t] + x[kp.CH["rain"]] * 20 - kc * x[kp.CH["et0"]] * 10 - sp
    assert float(kp.physics_residual(theta, h, X)) < 1e-6


def test_with_cut_hides_observations_only():
    s = ps.simulate(ps.WORLDS["prior"], 4, 6, INP)
    X = torch.as_tensor(s["X"])
    Xc = kp.with_cut(X, torch.tensor([40, 40, 91, 0]))
    assert Xc.shape[-1] == X.shape[-1] + 1
    assert float(Xc[0, 40:, kp.OBS_CH].abs().sum()) == 0 and float(Xc[0, 40:, -1].sum()) == 0
    assert torch.equal(Xc[2, :, :-1], X[2])
    assert torch.equal(Xc[0, :, kp.CH["et0"]], X[0, :, kp.CH["et0"]])  # weather stays visible


def test_answer_likelihood_and_weighted_quantiles():
    h = np.array([[-50.0, 10.0], [30.0, 60.0]])  # 2 samples x 2 fields
    ly = kp.answer_likelihood(h, np.array([1, 1]), "yesno")
    assert ly[0, 0] == pytest.approx(0.05) and ly[1, 0] == pytest.approx(0.95)
    ld = kp.answer_likelihood(h, np.array([2, 3]), "depth")
    assert ld[1, 0] > ld[0, 0] and ld[1, 1] > ld[0, 1]  # 30 mm fits "some", 60 mm fits "a lot"
    lg = kp.answer_likelihood(h, np.array([28.0, 0.0]), "gauge")
    assert lg[1, 0] > 0.9 and lg[0, 1] < 0.2 and lg[0, 0] < 1e-3
    vals = np.array([[1.0, 5.0], [2.0, 6.0], [3.0, 7.0]])
    w = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    q = kp.weighted_quantiles(vals, w, [0.5])
    assert q[0, 0] == 2.0 and q[0, 1] == 7.0


def test_fuse_pass_and_rates():
    rec, spec = kp.pass_rates(INP, "s2", np.array([0.2, 4.0]))
    assert rec[0] > 0.95 and rec[1] == 0.5  # open canopy sharp, closed canopy floored at chance
    p = kp.fuse_pass(np.array([0.5, 0.95, 0.5]), np.array([True, False, True]), np.array([0.9, 0.9, 0.5]),
                     np.array([0.9, 0.9, 0.5]))
    assert p[0] == pytest.approx(0.9)
    assert 0.6 < p[1] < 0.75       # a confident prior survives one contrary pass
    assert p[2] == pytest.approx(0.5)  # an uninformative pass changes nothing


def test_arai_particle_filter_runs_and_tracks():
    from awdrt import arai_baseline as ab
    rng = np.random.default_rng(0)
    w = np.array([[0.1, 0.2, 0.7], [1.0, 0.0, 0.0]])
    idx = ab._resample(w, rng)
    assert idx.shape == (2, 3) and (idx[1] == 0).all() and (idx[0] >= 0).all() and (idx[0] <= 2).all()
    s = ps.simulate(ps.WORLDS["prior"], 60, 7, INP)
    H = ps.end_of_day_depth(s["h6"])
    snaps, D = ab.run(s, INP, np.array([5.0, 0.03, 0.006, 0.25]), {55}, P=100)
    p_wet = np.nanmean(snaps[55]["h"] > 0, 1)
    ok = np.isfinite(H[:, 54])
    acc = np.mean((p_wet[ok] > 0.5) == (H[ok, 54] > 0))
    assert snaps[55]["h"].shape == (60, 100) and D.shape == (60, 100)
    assert acc > 0.6


def test_drop_logs_removes_only_log_channels():
    s = ps.simulate(ps.WORLDS["prior"], 6, 8, INP)
    X2 = kp.drop_logs(s["X"], np.array([True, False, True, False, False, False]))
    assert X2[[0, 2]][:, :, kp.LOG_CH].sum() == 0
    assert np.array_equal(X2[1], s["X"][1])
    other = [c for c in range(s["X"].shape[2]) if c not in kp.LOG_CH]
    assert np.array_equal(X2[:, :, other], s["X"][:, :, other])


def test_ablation_switches():
    s = ps.simulate(ps.WORLDS["prior"], 8, 9, INP)
    s["H"] = ps.end_of_day_depth(s["h6"])
    m = kp.PaddyNet(structure=False)
    th, st = kp.predict_v1(m, s["X"], s["static"], 91, n_samples=5)
    assert th.shape == (5, 8, 4) and np.all(th[..., 1:] == 0) and np.all(th[..., 0] > 0)
    Xs = kp.drop_satellite(s["X"])
    assert Xs[:, :, kp.SAT_CH].sum() == 0 and np.array_equal(Xs[:, :, kp.CH["et0"]], s["X"][:, :, kp.CH["et0"]])
    mt = kp.pretrain_v1(s, s, epochs=1, batch=4, structure=False, sat_off=True, phys_w=0.0, log=lambda _: None)
    assert not mt.structure
    lg = {"X": s["X"], "static": s["static"], "H_log": s["H"], "sp_log": s["sp_eff"]}
    kp.fine_tune_v1(mt, lg, s, steps=2, replay_batch=4)  # no-structure branch of fine-tuning


def test_fine_tune_regularisation_keeps_best_and_anchor():
    s = ps.simulate(ps.WORLDS["prior"], 8, 9, INP)
    s["H"] = ps.end_of_day_depth(s["h6"])
    lg = {"X": s["X"], "static": s["static"], "H_log": s["H"], "sp_log": s["sp_eff"]}
    m = kp.PaddyNet(structure=False)
    w0 = {k: v.clone() for k, v in m.state_dict().items()}
    # a validation score that only gets worse: early stopping keeps the starting weights
    calls = iter(range(100))
    kp.fine_tune_v1(m, lg, s, steps=4, replay_batch=4, val_fn=lambda _: next(calls), eval_every=2)
    assert all(torch.equal(w0[k], v) for k, v in m.state_dict().items()) and m.best_step_score == 0
    kp.fine_tune_v1(m, lg, s, steps=2, replay_batch=4, anchor_w=1e-3, lr_enc=0.0)
    assert all(torch.equal(w0[k], v) for k, v in m.state_dict().items() if k.startswith(("inp.", "gru.")))


def test_boro_phenology():
    """Experiment 27's crop: slow in January, fast through tillering, canopy closed near heading; NISAR loses
    accuracy above 70 cm. The module default stays the straight line of Experiments 1-26."""
    import numpy as np
    from awdrt import paddy_sim as ps
    inp = ps.load_inputs()
    old = ps.PHENOLOGY
    try:
        ps.set_phenology("boro")
        ps.set_temperature(inp["tmean"])
        tp = float(inp["s0"])                      # transplanted on 15 January
        c = ps.crop_state(tp + np.array([0, 30, 60, 90]), tp)
        assert c["lai"][0] < 0.5 and c["lai"][1] < 1.0 and 3.0 < c["lai"][2] < 4.5 and c["lai"][3] > 5.0
        assert np.all(np.diff(c["height_cm"]) > 0) and 15 < c["height_cm"][0] < 25 and c["height_cm"][3] > 85
        till = ps.crop_state(tp + np.arange(0, 100), tp)["tillers"]
        assert till.max() > 20 and till[-1] < till.max() - 3     # tillers peak, then some die back
        late = ps.crop_state(tp + 30.0, tp - 20.0)["lai"]       # transplanted earlier: further on
        assert late > c["lai"][1]
        assert ps.nisar_accuracy(0.8, 60.0) == 0.8 and abs(ps.nisar_accuracy(0.8, 100.0) - 0.65) < 1e-9
        ps.set_phenology("linear")
        assert abs(float(ps.lai_curve(tp + 40, tp)) - 2.0) < 1e-9 and ps.nisar_accuracy(0.8, 100.0) == 0.8
    finally:
        ps.set_phenology(old)
