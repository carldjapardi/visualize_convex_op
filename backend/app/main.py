from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .methods import list_method_summaries, solve
from .problems import get_problem, list_problem_summaries, sample_geometry
from .schemas import GeometryRequest, GeometryResponse, MethodSummary, ProblemSummary, SolveRequest, SolveResponse


app = FastAPI(
    title="Optimization Visualizer API",
    description="Solve and visualize LP, QP, QCQP, SOCP, and SDP examples with educational solvers.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/problem-types", response_model=list[ProblemSummary])
def problem_types() -> list[ProblemSummary]:
    return list_problem_summaries()


@app.get("/methods", response_model=list[MethodSummary])
def methods() -> list[MethodSummary]:
    return list_method_summaries()


@app.post("/sample-geometry", response_model=GeometryResponse)
def geometry(request: GeometryRequest) -> GeometryResponse:
    try:
        get_problem(request.problem_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return sample_geometry(request.problem_id, request.resolution)


@app.post("/solve", response_model=SolveResponse)
def solve_problem(request: SolveRequest) -> SolveResponse:
    try:
        return solve(request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
