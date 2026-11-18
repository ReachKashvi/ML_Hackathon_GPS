"""PathSnap FastAPI backend (Laptop 1).

Endpoints (the contract ``ui/api_client.py`` expects):

``GET /health``           ``{"status": "ok", "model": ..., "config": {...}}``
``GET /trips``            ``{"trips": ["00000000", ...]}`` -- the trips in ``data_dir``
``GET /predict/{trip_id}`` matched route, GPS, nearby road network and the
                          server-side metrics (Top-1, continuity, counts, time)

Run from the repository root::

    python -m uvicorn app:app --app-dir api --host 0.0.0.0 --port 8000

Every prediction is also logged to this console, so the receiving laptop
shows each request and its result.
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from inference import Predictor

log = logging.getLogger("pathsnap")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")

predictor = Predictor()
app = FastAPI(title="PathSnap map-matching API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])

log.info("model loaded: %s", predictor.model_name)
log.info("serving %d trips from %s", len(predictor.list_trips()), predictor.data_dir)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": predictor.model_name, "config": predictor.describe()}


@app.get("/trips")
def trips() -> dict:
    return {"trips": predictor.list_trips()}


@app.get("/predict/{trip_id}")
def predict(trip_id: str) -> dict:
    t0 = time.perf_counter()
    try:
        out = predictor.predict(trip_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown trip {trip_id!r}")
    except Exception as exc:  # surface the reason to the evaluator UI
        log.exception("prediction failed for %s", trip_id)
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
    m = out["metrics"]
    top1 = "n/a" if m["top1_accuracy"] is None else f"{m['top1_accuracy']:.4f}"
    log.info("predict %s  points=%d  arcs=%d  top1=%s  continuity=%.4f  "
             "inference=%.2fs  request=%.2fs",
             trip_id, m["n_gps_points"], m["n_predicted_arcs"], top1,
             m["route_continuity"] or 0.0, m["inference_time_sec"], time.perf_counter() - t0)
    return out
