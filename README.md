# Optimization Visualizer

Interactive visualizations for LPs, QPs, QCQPs, SOCPs, and SDPs. The frontend is a React/Vite app, and the backend is a FastAPI service that uses SciPy and CVXPY for real numerical solver outputs.

## What It Shows

- Feasible regions and objective contours for 2D examples.
- Solver solutions for LP, QP, QCQP, SOCP, and SDP examples.
- Method traces for GD, SGD, PGD, and Newton's method where those methods are meaningful.
- Simplex and interior-point style results through SciPy/CVXPY solver backends.
- SDP diagnostics through a trace-one 2x2 PSD slice and eigenvalue output.

## Setup

Create and install the Python backend:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e backend
```

Install the frontend:

```bash
cd frontend
npm install
```

## Run Locally

Start the backend:

```bash
source .venv/bin/activate
uvicorn app.main:app --app-dir backend --reload
```

Start the frontend in another terminal:

```bash
cd frontend
npm run dev
```

Open `http://localhost:5173`.

## API

- `GET /problem-types` lists curated LP, QP, QCQP, SOCP, and SDP examples.
- `GET /methods` lists available numerical methods.
- `POST /solve` solves a selected problem/method pair.
- `POST /sample-geometry` returns plot-ready feasible region and contour data.

## Notes

Not every method is meaningful for every problem class. The app exposes compatibility explicitly: Simplex is LP-specific, interior-point/conic solvers cover the conic examples, and GD/SGD/PGD/Newton traces are used for smooth low-dimensional demos.
