# Paddy Twin: KGML-Paddy research simulation as a web app (CPU only).
# Build from the repository root:  docker compose up --build   (or: docker build -f app/Dockerfile -t paddy-twin .)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TORCH_THREADS=2
WORKDIR /srv

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY app/requirements.txt app/requirements.txt
RUN pip install -r app/requirements.txt

# only what the simulator and model read: the library, the trained weights and the input data
COPY src/awdrt/__init__.py src/awdrt/paddy_sim.py src/awdrt/kgml_paddy.py src/awdrt/counterfactual.py src/awdrt/methane.py src/awdrt/
COPY results/models/kgml_paddy_v1b_pretrained.pt results/models/kgml_paddy_v1c_pretrained.pt results/models/kgml_paddy_v1d_pretrained.pt results/models/
COPY results/era5_daily_jamalpur.csv results/real_validation_metrics.json results/real_season_curve.csv \
     results/simulation_metrics.json results/s1_wetdry_accuracy.json results/
COPY app/__init__.py app/engine.py app/server.py app/sites.py app/campaign.py app/qc.py app/planner.py app/experiment.py app/
COPY app/sites app/sites
COPY app/campaigns app/campaigns
COPY app/static app/static

RUN useradd --create-home twin
USER twin
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
