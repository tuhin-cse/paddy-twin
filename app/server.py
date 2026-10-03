"""Paddy Twin web server: FastAPI around app/engine.py, serving the single-page front end.

Run locally:  uvicorn app.server:app --port 8000   (from the repository root)
Docker:       docker compose up --build            (then open http://localhost:8000)
"""

import asyncio
import os
import pathlib
import threading
import uuid
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import engine, experiment, planner

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))
STATIC = pathlib.Path(__file__).with_name("static")
app = FastAPI(title="Paddy Twin", description="KGML-Paddy research simulation (simulated data only)")
_inp = engine.ps.load_inputs()
_lock = asyncio.Lock()  # one season at a time: the model run uses all threads


class Ask(BaseModel):
    run: str
    field: int
    day: int
    kind: str = "yesno"
    answer: int | None = None


@app.get("/api/defaults")
def defaults():
    return {"defaults": engine.DEFAULTS, "limits": engine.LIMITS, "worlds": list(engine.ps.WORLDS)}


@app.get("/api/sites")
def site_list():
    """Field layouts: the generated grid and every GeoJSON site file in app/sites/."""
    return engine.sites.list_sites()


@app.post("/api/season")
async def season(params: dict):
    token = str(params.pop("_token", "") or "")[:40] or None
    async with _lock:
        out = await asyncio.to_thread(engine.run_season, params, _inp, token)
    engine.progress_done(token)
    return out


@app.get("/api/progress/{token}")
def progress(token: str):
    """Where a season or replay run started with this token is: its stages, the current one, and the fraction done.
    Before the run starts (it may wait for another one), the token is unknown: 'waiting'."""
    return engine.PROGRESS.get(token) or {"waiting": True}


@app.get("/api/campaigns")
def campaign_list():
    """Campaign datasets in app/campaigns/ (replay mode)."""
    return engine.campaign.list_campaigns()


@app.post("/api/replay")
async def replay(params: dict):
    name = str(params.get("campaign", ""))
    token = str(params.pop("_token", "") or "")[:40] or None
    async with _lock:
        try:
            out = await asyncio.to_thread(engine.run_replay, name, params, _inp, token)
            engine.progress_done(token)
            return out
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        except (ValueError, OSError) as e:
            raise HTTPException(400, f"campaign {name}: {e}") from e


@app.post("/api/ask")
async def ask(q: Ask):
    if q.kind not in ("yesno", "depth", "none"):
        raise HTTPException(400, "kind must be yesno, depth or none")
    try:
        return await asyncio.to_thread(engine.ask_farmer, q.run, q.field, q.day, q.kind, q.answer)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    except (ValueError, IndexError) as e:
        raise HTTPException(400, str(e)) from e


@app.get("/api/qc/{run}/report.md")
def qc_report(run: str, week: int | None = None):
    """The weekly data-quality report of a replay run (Markdown; week = index into qc.weeks, default the last)."""
    try:
        text = engine.qc_report(run, "md", week)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return PlainTextResponse(text, media_type="text/markdown; charset=utf-8",
                             headers={"content-disposition": f'attachment; filename="qc_report_{run}.md"'})


@app.get("/api/qc/{run}/issues.csv")
def qc_issues(run: str):
    """Every data-quality issue of a replay run as CSV."""
    try:
        text = engine.qc_report(run, "csv")
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    return PlainTextResponse(text, media_type="text/csv; charset=utf-8",
                             headers={"content-disposition": f'attachment; filename="qc_issues_{run}.csv"'})


class Plan(BaseModel):
    site: str = "grid"
    n_fields: int = 120
    fields: list[int] = []
    n: int = 40
    locked: list[int] = []
    excluded: list[int] = []
    p_fail: float = 0.1
    pairs_target: int = 10
    format: str = "csv"


def _plan_layout(q: Plan):
    if q.site == "grid" and not engine.LIMITS["n_fields"][0] <= q.n_fields <= engine.LIMITS["n_fields"][1]:
        raise HTTPException(400, "n_fields out of range")
    if not 0.0 <= q.p_fail <= 0.6 or not 0 <= q.pairs_target <= 50:
        raise HTTPException(400, "p_fail must be 0-0.6 and pairs_target 0-50")
    try:
        return planner.get_layout(q.site, q.n_fields)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.post("/api/plan/site")
async def plan_site(q: Plan):
    """The planner's view of a site: strata per plot, eligibility, pipe positions, adjacent plots."""
    lay = _plan_layout(q)
    return await asyncio.to_thread(planner.site_summary, lay)


@app.post("/api/plan/evaluate")
async def plan_evaluate(q: Plan):
    lay = _plan_layout(q)
    return await asyncio.to_thread(planner.evaluate, lay, q.fields, q.p_fail, q.pairs_target)


@app.post("/api/plan/suggest")
async def plan_suggest(q: Plan):
    """A suggested plan of q.n plots (keeping q.locked, never using q.excluded or plots without consent)."""
    lay = _plan_layout(q)
    sel = await asyncio.to_thread(planner.suggest, lay, q.n, q.locked, q.excluded, q.pairs_target)
    return await asyncio.to_thread(planner.evaluate, lay, sel, q.p_fail, q.pairs_target)


@app.post("/api/plan/export")
async def plan_export(q: Plan):
    """The plan as CSV (the installation list, in visiting order) or as a site GeoJSON with the planned loggers."""
    lay = _plan_layout(q)
    if q.format == "csv":
        text = await asyncio.to_thread(planner.export_csv, lay, q.fields)
        return PlainTextResponse(text, media_type="text/csv; charset=utf-8",
                                 headers={"content-disposition": f'attachment; filename="logger_plan_{q.site}.csv"'})
    if q.format == "geojson":
        try:
            text = planner.export_geojson(lay, q.fields, None if q.site == "grid" else engine.sites.SITES_DIR / f"{q.site}.geojson")
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return PlainTextResponse(text, media_type="application/geo+json",
                                 headers={"content-disposition": f'attachment; filename="{q.site}_plan.geojson"'})
    raise HTTPException(400, "format must be csv or geojson")


class Counterfactual(BaseModel):
    run: str
    wait: float = 0
    risk: float = 0.5
    lead: float = 1
    follow: float = 1.0
    ask: bool = True


@app.post("/api/counterfactual")
async def counterfactual(q: Counterfactual):
    """The season's village re-run under each irrigation policy: the farmers' practice, KGML-Paddy's advice,
    the same advice with a regional or no S&P, and advice from perfect knowledge."""
    async with _lock:
        try:
            return await asyncio.to_thread(engine.run_counterfactual, q.run, q.model_dump(exclude={"run"}))
        except KeyError as e:
            raise HTTPException(404, str(e)) from e


class Methane(BaseModel):
    run: str
    min_dry: int = 2
    regional: str = "aggregated"


@app.post("/api/methane")
def methane(q: Methane):
    """Each field's water regime and IPCC Tier 1 methane, from the truth or loggers, the model, the satellite
    calls alone, and one regional default."""
    try:
        return engine.run_methane(q.run, q.model_dump(exclude={"run"}))
    except KeyError as e:
        raise HTTPException(404, str(e)) from e


_jobs = {}            # experiment id -> {status, done, total, seconds, result | error}
_job_lock = threading.Lock()


class ExperimentSpec(BaseModel):
    base: dict = {}
    vary: str = "nisar_update"
    a: Any = False
    b: Any = True
    villages: int = 5
    seed0: int = 1
    methane: bool = False
    counterfactual: bool = False
    cf: dict = {}


@app.post("/api/experiment")
def experiment_start(q: ExperimentSpec):
    """Start an experiment (one comparison over several villages) in the background; poll GET /api/experiment/<id>."""
    with _job_lock:
        if any(j["status"] == "running" for j in _jobs.values()):
            raise HTTPException(409, "an experiment is already running")
        jid = uuid.uuid4().hex[:10]
        _jobs[jid] = {"status": "running", "done": 0, "total": 2 * min(max(q.villages, 2), experiment.MAX_VILLAGES), "seconds": 0}
        while len(_jobs) > 10:
            _jobs.pop(next(iter(_jobs)))
    spec = q.model_dump()

    def work():
        def progress(done, total, secs):
            _jobs[jid].update(done=done, total=total, seconds=round(secs, 1))
        try:
            _jobs[jid]["result"] = experiment.run(spec, progress)
            _jobs[jid]["status"] = "done"
        except Exception as e:  # reported to the page
            _jobs[jid].update(status="failed", error=str(e))

    threading.Thread(target=work, daemon=True).start()
    return {"id": jid}


@app.get("/api/experiment/{jid}")
def experiment_status(jid: str):
    if jid not in _jobs:
        raise HTTPException(404, "unknown experiment")
    return _jobs[jid]


@app.get("/api/experiment/{jid}/export.{fmt}")
def experiment_export(jid: str, fmt: str):
    j = _jobs.get(jid)
    if not j or j["status"] != "done":
        raise HTTPException(404, "no finished experiment with this id")
    if fmt == "csv":
        return PlainTextResponse(experiment.to_csv(j["result"]), media_type="text/csv; charset=utf-8",
                                 headers={"content-disposition": f'attachment; filename="experiment_{jid}.csv"'})
    if fmt == "md":
        return PlainTextResponse(experiment.to_markdown(j["result"]), media_type="text/markdown; charset=utf-8",
                                 headers={"content-disposition": f'attachment; filename="experiment_{jid}.md"'})
    raise HTTPException(400, "format must be csv or md")


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
if engine.sites.IMAGERY_DIR.is_dir():   # real Sentinel-2 images under the 3D village (scripts/fetch_site_imagery.py)
    app.mount("/imagery", StaticFiles(directory=engine.sites.IMAGERY_DIR), name="imagery")
