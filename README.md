# Paddy Twin: learning each rice field's water loss from satellites and farmer logs

KGML-Paddy is a knowledge-guided model that learns each paddy field's seepage + percolation (S&P) and daily
water depth from satellite ponding timing, one-tap farmer irrigation logs and tubewell pumping totals. It is
pre-trained on a paddy simulator and fine-tuned on field loggers. **Paddy Twin** is an interactive digital
twin of a village of rice fields through one Boro season, running the simulator and the model live.

**Every number in the app is simulation, not field data.**

## Layout

```
app/           Paddy Twin web app (FastAPI + single-page front end; Docker), see app/README.md
src/awdrt/     library: PROSAIL + water background (rtm, backgrounds, s2), paddy water balance (hydro),
               simulator v2 (paddy_sim), KGML-Paddy model (kgml_paddy), Arai-style baseline (arai_baseline),
               closed-loop advice (counterfactual), IPCC Tier 1 methane (methane), ERA5 (era5),
               data access (data_sources), .env loading (env)
tests/         fast checks of the simulator, the model and the app (python -m pytest tests)
scripts/       numbered experiments 01–27; each writes to results/
results/       metrics (JSON/CSV), figures and trained model weights
data/cache/    downloaded scenes and simulation caches (git-ignored)
```

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build is enough
pip install -r requirements.txt h5py fsspec aiohttp requests
python scripts/01_simulation_experiment.py     # Layer 1 feasibility, offline, ~2 min
python scripts/08_simulator_mismatch.py        # simulator mismatch, ~15 min
python scripts/09_kgml_paddy_v0.py             # KGML-Paddy v0 (PyTorch), ~40 min
python scripts/11_kgml_paddy_v1_state.py       # v1: daily water level + next-dry forecast, ~8 min
python -m pytest tests                         # fast checks
```

## Run the app

```bash
docker compose up --build        # then open http://localhost:8000
```

## Experiments

| Script | What it tests |
| --- | --- |
| `01` | PROSAIL: can Sentinel-2 tell ponded, saturated and drying backgrounds under a rice canopy |
| `02` | Real Sentinel-2 Boro season over Jamalpur |
| `03`, `03b` | Sentinel-1 fusion and Sentinel-1 wet/dry accuracy |
| `04`–`04e` | Field leakage twin, ERA5 ET0, simulation-based inference, top-up behaviour, app-only ablation, revisit and sparse loggers |
| `05`, `06` | SoilGrids check, ET0 cross-check |
| `08` | Pre-train on a simple simulator, test on richer physics: how many loggers fine-tuning needs |
| `09` | Simulator v2 and KGML-Paddy v0: pre-training, fine-tuning through the physics, conformal calibration |
| `10` | Gradient-boosted tree baseline on the same data |
| `11`, `12` | KGML-Paddy v1: daily water level, next-dry forecast, fine-tuning the water-level head |
| `13` | Value of a farmer's "is there water now?" answer |
| `14` | N3 in real time: water-balance check of satellite passes |
| `15` | Arai-style particle-filter baseline |
| `16`, `17` | Tubewell meter totals and training with non-logging farmers (v1b) |
| `18`, `18b` | Full ablation |
| `19` | How many logger fields the calibrations need |
| `20`, `21`, `22` | S&P law extrapolation, regularised fine-tuning, wider S&P prior (v1c) |
| `23`, `24` | NISAR as a model input (v1n) and as an answer update |
| `25` | Advice against practice in closed loop |
| `26` | Field water regimes and IPCC Tier 1 methane |
| `27` | Realistic Boro rice growth in thermal time (v1d, the app's default) |

## Third-party assets

three.js r170 (MIT, `app/static/vendor/three/LICENSE`) and Noto Sans Bengali (SIL OFL,
`app/static/vendor/fonts/OFL.txt`) are vendored so the app runs offline.
