"""Telemetry schemas and endpoints."""
from fastapi import APIRouter, Depends, Query

from app.api.deps import get_repos
from app.api.guards import require_user
from app.infrastructure.model_adapter import Frame, get_adapter

router = APIRouter(prefix="/api/runs", tags=["telemetry"])


@router.get("/{run_id}/telemetry", summary="Telemetry frames for a run",
            responses={404: {"description": "Run not found"}})
def telemetry(
    run_id: str,
    from_t: float = Query(0.0, ge=0.0, alias="from"),
    to_t: float = Query(1e9, ge=0.0, alias="to"),
    step: int = Query(1, ge=1, le=100),
    _: None = Depends(require_user),
):
    frames = list(get_adapter().replay(run_id))
    if not frames:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Run not found")
    frames = [f for f in frames if from_t <= f.t <= to_t]
    return [f.__dict__ for f in frames[::step]]


@router.get("/{run_id}/events", summary="Events (slip, trust drops, etc.)")
def events(run_id: str, _: None = Depends(require_user)):
    frames = list(get_adapter().replay(run_id))
    if not frames:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Run not found")
    ev = []
    for f in frames:
        if f.trust < 0.5 or f.slip > 0.3:
            ev.append({"t": f.t, "type": "slip" if f.slip > 0.3 else "low_trust",
                       "desc": f"trust={f.trust:.2f}, slip={f.slip:.2f}"})
    return ev


@router.get("/{run_id}/report", summary="Summary metrics for a run")
def report(run_id: str, _: None = Depends(require_user)):
    frames = list(get_adapter().replay(run_id))
    if not frames:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Run not found")
    n = len(frames)
    v_mod = sum(f.v_model for f in frames) / n
    v_gns = sum(f.v_gnss or 0 for f in frames) / n
    v_whl = sum(f.v_wheels for f in frames) / n
    v_err_model = (sum((f.v_model - (f.v_gnss or f.v_model))**2 for f in frames) / n) ** 0.5
    v_err_wheels = (sum((f.v_wheels - (f.v_gnss or f.v_wheels))**2 for f in frames) / n) ** 0.5
    return {
        "run_id": run_id,
        "frames": n,
        "duration": frames[-1].t if frames else 0.0,
        "avg_speed_model": v_mod,
        "avg_speed_gnss": v_gns,
        "avg_speed_wheels": v_whl,
        "rmse_model": v_err_model,
        "rmse_wheels": v_err_wheels,
        "win": v_err_wheels - v_err_model,
    }