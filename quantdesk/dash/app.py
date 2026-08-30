"""Research review dashboard.

    python -m quantdesk.dash.app        ->  http://127.0.0.1:8010

Structure lifted from `nyam_bias/app.py` - FastAPI + Jinja + vanilla JS, no
build step - with the pre-market bias analysis removed and the backtest honesty
panels put in its place.

WHAT IT SHOWS, AND WHY THAT ORDER
---------------------------------
1. The verdict, with its reasons.
2. Gross beside net. The gap between them is the finding, so they never appear
   on separate screens.
3. Cost sensitivity - how wrong the cost estimate can be before the edge dies.
4. Walk-forward, window by window. One profitable window out of nine is visible
   here and invisible in an equity curve.
5. The equity curve, last. It is the prettiest panel and the least informative,
   so it does not get to lead.

The dashboard computes nothing. Every figure is read from a run JSON written by
`backtest/`, so there is no second implementation that can drift from the one
that produced the grade.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from ..core.clock import Timeframe
from ..data.store import BarStore
from .results import RUNS_DIR, list_runs, load_run

HERE = Path(__file__).parent
ROOT = HERE.parents[1]

app = FastAPI(title="quantdesk research review")
app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")
templates = Jinja2Templates(directory=str(HERE / "templates"))


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "dashboard.html", {"request": request})


@app.get("/api/runs")
def api_runs() -> JSONResponse:
    return JSONResponse(list_runs(ROOT / RUNS_DIR))


@app.get("/api/run/{run_id}")
def api_run(run_id: str) -> JSONResponse:
    try:
        return JSONResponse(load_run(run_id, ROOT / RUNS_DIR))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/health")
def api_health() -> JSONResponse:
    """Data-layer health. Rendered as a panel because a stale store is the
    quietest way for a backtest to become meaningless."""
    report: dict[str, object] = {"store": None, "runs": 0, "errors": []}
    report["runs"] = len(list_runs(ROOT / RUNS_DIR))

    store_path = ROOT / "data" / "demo_bars.sqlite"
    if not store_path.exists():
        report["errors"].append(f"no bar store at {store_path.name}")
        return JSONResponse(report)

    try:
        with BarStore(store_path) as store:
            series = []
            for symbol, tf_label in store.symbols():
                timeframe = Timeframe.parse(tf_label)
                coverage = store.coverage(symbol, timeframe)
                if coverage is None:
                    continue
                first, last, count = coverage
                series.append({
                    "symbol": symbol,
                    "timeframe": tf_label,
                    "first": first.isoformat(),
                    "last": last.isoformat(),
                    "bars": count,
                })
            report["store"] = {"path": store_path.name, "series": series}
    except Exception as exc:  # a broken store must be visible, not silent
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    return JSONResponse(report)


def main(host: str = "127.0.0.1", port: int = 8010) -> None:
    import uvicorn

    print(f"quantdesk research review -> http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
