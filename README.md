# Optimization Visualizer

An interactive lab for exploring LPs, QPs, QCQPs (quadratically constrained quadratic programs), SOCPs, and SDPs. The frontend is a React/Vite app; the backend is FastAPI with **in-repo educational methods** (no SciPy or CVXPY).

## What It Shows

- **25 curated 2D examples** — 5 each for LP, QP, QCQP, SOCP, and SDP.
- Feasible regions, objective contours, a value map, and a rotatable **3D objective surface**.
- Step-by-step paths with an iteration scrubber and objective/violation history.
- A guided objective and learning goal for each example.

These are fixed two-dimensional teaching examples, not a general-purpose optimization solver. The "Constrained Reference" method uses a log-barrier path and exact corner refinement for LPs; other classes use projected steps and sampled feasible refinement, so their results are labeled **approximate**. SDP plots show a trace-one slice of 2×2 positive-semidefinite matrices.

## Example Catalog (5 per class)

### LP — Linear programs

| ID | Name | Methods |
|----|------|---------|
| `lp-basic` | LP: Production Mix | simplex, interior_point |
| `lp-warehouse` | LP: Warehouse Allocation | simplex, interior_point |
| `lp-diet` | LP: Diet Planning | simplex, interior_point |
| `lp-transport` | LP: Transport Balance | simplex, interior_point |
| `lp-max-flow-slice` | LP: Max Flow Slice | simplex, interior_point |

### QP — Quadratic programs

| ID | Name | Methods |
|----|------|---------|
| `qp-constrained` | QP: Constrained Quadratic Bowl | interior_point, gd, sgd, pgd, newton |
| `qp-portfolio` | QP: Portfolio Variance Tradeoff | interior_point, gd, sgd, pgd, newton |
| `qp-least-squares` | QP: Least Squares Box | interior_point, gd, sgd, pgd, newton |
| `qp-elastic-net-slice` | QP: Anisotropic Quadratic | interior_point, gd, sgd, pgd, newton |
| `qp-saddle-slice` | QP: Strongly Convex Slice | interior_point, gd, sgd, pgd, newton |

### QCQP — Quadratically constrained (QOCP)

| ID | Name | Methods |
|----|------|---------|
| `qcqp-intersection` | QCQP: Quadratic Constraint Intersection | interior_point, gd, sgd, pgd, newton |
| `qcqp-disk-cap` | QCQP: Disk With Linear Cap | interior_point, gd, sgd, pgd, newton |
| `qcqp-trust-region` | QCQP: Trust Region | interior_point, pgd |
| `qcqp-ellipse-box` | QCQP: Ellipse and Box | interior_point, gd, sgd, pgd, newton |
| `qcqp-naive-gd` | QCQP: Naive GD Demo | interior_point, gd, sgd |

### SOCP — Second-order cone programs

| ID | Name | Methods |
|----|------|---------|
| `socp-cone` | SOCP: Tilted Norm Bound | interior_point |
| `socp-robust-line` | SOCP: Affine Norm Budget | interior_point |
| `socp-hyperbolic` | SOCP: Steep Cone Section | interior_point |
| `socp-norm-ball-intersect` | SOCP: Norm Ball and Box | interior_point |
| `socp-portfolio-risk` | SOCP: Linear Reward and Norm Budget | interior_point |

### SDP — Semidefinite programs (2×2 trace-one slice)

| ID | Name | Methods |
|----|------|---------|
| `sdp-trace` | SDP: 2x2 PSD Trace Slice | interior_point |
| `sdp-max-cut-slice` | SDP: Objective Tilt | interior_point |
| `sdp-dual-slice` | SDP: Eigenvalue Check | interior_point |
| `sdp-narrow-cone` | SDP: Boundary Zoom | interior_point |
| `sdp-wide-cone` | SDP: Interior Start | interior_point |

## Methods vs Problem Classes

| Method | LP | QP | QCQP | SOCP | SDP |
|--------|:--:|:--:|:----:|:----:|:---:|
| Simplex | yes | — | — | — | — |
| Constrained Reference (`interior_point`) | yes | yes | yes | yes | yes |
| GD / SGD | — | yes | some* | — | — |
| PGD | — | yes | yes | — | — |
| Newton | — | yes | yes** | — | — |

\*Example `qcqp-naive-gd` omits PGD on purpose to show GD leaving the feasible set.

\*\*Convex smooth QCQPs only.

The UI only lists methods compatible with the selected example.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e "backend[dev]"
```

```bash
cd frontend
npm install
```

## Run Locally

```bash
source .venv/bin/activate
uvicorn app.main:app --app-dir backend --reload
```

```bash
cd frontend
npm run dev
```

Open `http://localhost:5173`.

## API

- `GET /problem-types` — 25 problem summaries.
- `GET /methods` — method metadata.
- `POST /solve` — run an educational solver.
- `POST /sample-geometry` — plot grids and boundaries.

## Tests

```bash
source .venv/bin/activate
pytest backend/tests -q
```

## Problem Hierarchy

`LP ⊂ QP ⊂ QCQP ⊂ SOCP ⊂ SDP` (convex tractable classes). QCQP is the same class often called **QOCP** (quadratic objective and quadratic constraints).
