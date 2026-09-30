# Optimization Visualizer

An interactive lab for exploring five convex optimization families and a separate family of nonconvex functions. The frontend is a React/Vite app; the backend is FastAPI with **in-repo educational methods** (no SciPy or CVXPY).

## What It Shows

- **23 curated 2D examples** across LP, QP, QCQP, SOCP, SDP, and nonconvex landscapes. Repeated cases have been removed from the catalog.
- Feasible regions, objective contours, a value map, and a rotatable **3D objective surface**. QP, QCQP, and nonconvex examples open on the surface.
- Step-by-step paths with an iteration scrubber and objective/violation history.
- A guided objective and learning goal for each example, family and method formulas, and interactive method parameters. Nonconvex examples also let you move the starting point.

These are fixed two-dimensional teaching examples, not a general-purpose optimization solver. The "Constrained Reference" method uses a log-barrier path and exact corner refinement for LPs; other convex classes use projected steps and sampled feasible refinement, so their results are labeled **approximate**. SDP plots show a trace-one slice of 2×2 positive-semidefinite matrices. Nonconvex paths are local searches; they do not certify a global minimum.

## Example Catalog

### LP — Linear programs

| ID | Name | Methods |
|----|------|---------|
| `lp-basic` | LP: Production Mix | simplex, interior_point |
| `lp-ratio-blend` | LP: Ratio-Constrained Blend | simplex, interior_point |
| `lp-diet` | LP: Diet Planning | simplex, interior_point |
| `lp-delivery-window` | LP: Delivery Window | simplex, interior_point |
| `lp-alternate-optima` | LP: Alternate Optima | simplex, interior_point |

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

### SOCP — Second-order cone programs

| ID | Name | Methods |
|----|------|---------|
| `socp-cone` | SOCP: Tilted Norm Bound | interior_point |
| `socp-hyperbolic` | SOCP: Steep Cone Section | interior_point |
| `socp-norm-ball-intersect` | SOCP: Norm Ball and Box | interior_point |

### SDP — Semidefinite programs (2×2 trace-one slice)

| ID | Name | Methods |
|----|------|---------|
| `sdp-trace` | SDP: 2x2 PSD Trace Slice | interior_point |
| `sdp-max-cut-slice` | SDP: Objective Tilt | interior_point |
| `sdp-dual-slice` | SDP: Eigenvalue Check | interior_point |

### Nonconvex functions (3 local-search landscapes)

| ID | Name | Methods |
|----|------|---------|
| `nonconvex-double-well` | Tilted Double Well | gd, sgd, newton |
| `nonconvex-himmelblau` | Four Basins | gd, sgd, newton |
| `nonconvex-rippled-bowl` | Rippled Bowl | gd, sgd, newton |

## Methods vs Problem Classes

| Method | LP | QP | QCQP | SOCP | SDP | Nonconvex |
|--------|:--:|:--:|:----:|:----:|:---:|:---------:|
| Simplex | yes | — | — | — | — | — |
| Constrained Reference (`interior_point`) | yes | yes | yes | yes | yes | — |
| GD / SGD | — | yes | some* | — | — | yes |
| PGD | — | yes | yes | — | — | — |
| Newton | — | yes | yes** | — | — | yes |

\*The QCQP Trust Region example uses PGD only; on Quadratic Constraint Intersection, compare GD leaving the feasible set with PGD staying inside.

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

- `GET /problem-types` — 23 problem summaries.
- `GET /methods` — method metadata.
- `POST /solve` — run an educational solver.
- `POST /sample-geometry` — plot grids and boundaries.

## Tests

```bash
source .venv/bin/activate
pytest backend/tests -q
```

## Problem Hierarchy

`LP ⊂ QP ⊂ QCQP ⊂ SOCP ⊂ SDP` describes the convex classes represented here. The nonconvex landscapes are separate. QCQP is the same class often called **QOCP** (quadratic objective and quadratic constraints).
