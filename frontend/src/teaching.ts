import type { MethodId, ProblemType } from "./types";

export const FAMILY_GUIDES: Record<ProblemType, { name: string; short: string; form: string; explanation: string; focus: string }> = {
  lp: {
    name: "Linear programming", short: "Straight boundaries and a linear objective",
    form: "min cᵀx   subject to Ax ≤ b",
    explanation: "Both the objective and every constraint are linear. In two variables, the feasible set is a polygon; a bounded optimum can be found at a corner, although an entire edge can tie.",
    focus: "Watch which constraints become active at the final point.",
  },
  qp: {
    name: "Convex quadratic programming", short: "A curved objective with linear limits",
    form: "min ½xᵀQx + cᵀx   with Q ⪰ 0",
    explanation: "A positive semidefinite quadratic makes bowl-shaped level sets. Linear constraints can push the minimum from the bowl's center to a boundary.",
    focus: "Compare descent directions with the shape of the contours.",
  },
  qcqp: {
    name: "Convex quadratically constrained programming", short: "Curved objectives and curved limits",
    form: "min f(x)   subject to gᵢ(x) ≤ 0",
    explanation: "The examples use convex quadratic objectives and convex quadratic inequalities. Their feasible boundaries may be disks or ellipses rather than straight lines.",
    focus: "See whether an unconstrained step crosses a curved boundary.",
  },
  socp: {
    name: "Second-order cone programming", short: "Norm constraints and cone sections",
    form: "min cᵀx   subject to ‖Ax + b‖₂ ≤ dᵀx + e",
    explanation: "A norm is bounded by an affine expression. The curved region shown here is a two-variable section of a second-order cone constraint.",
    focus: "Look for the active norm boundary at the solution.",
  },
  sdp: {
    name: "Semidefinite programming", short: "A two-dimensional slice of PSD matrices",
    form: "min C • X   subject to X ⪰ 0",
    explanation: "The variable is a matrix whose eigenvalues must stay nonnegative. This site shows the trace-one 2×2 slice X = [[a,b],[b,1−a]].",
    focus: "The curved boundary is where the smaller eigenvalue reaches zero.",
  },
  nonconvex: {
    name: "Nonconvex landscapes", short: "Several valleys and local stationary points",
    form: "min f(x,y)   with Hessian not always ⪰ 0",
    explanation: "These objectives can have multiple valleys, saddles, and locally optimal points. A descent path depends on its starting point; reaching a stationary point does not certify a global minimum.",
    focus: "Move the start and compare where the same method ends.",
  },
};

export function methodGuide(method: MethodId, family: ProblemType): { formula: string; explanation: string; parameter: string } {
  if (method === "simplex") return {
    formula: "vₖ₊₁ = argmin { cᵀv : v is an improving neighbor of vₖ }",
    explanation: "Move along a feasible polygon edge to a better adjacent vertex. Stop when no adjacent vertex improves the objective.",
    parameter: "No step size: each pivot goes from one corner to another.",
  };
  if (method === "interior_point") {
    if (family === "lp") return {
      formula: "x(μ) = argmin [ cᵀx − μ Σ log sᵢ(x) ]",
      explanation: "The positive slacks sᵢ keep the path inside the LP. As μ shrinks, the path approaches the boundary; an exact corner check finishes this 2D example.",
      parameter: "μ is reduced internally; the final LP corner is exact.",
    };
    if (family === "qp") return {
      formula: "xₖ₊₁ = ΠC(xₖ − H⁻¹∇f(xₖ))",
      explanation: "A Newton-like step uses quadratic curvature, then projects into the feasible set. A sampled feasible check may improve the final reference.",
      parameter: "This educational reference is approximate.",
    };
    if (family === "qcqp") return {
      formula: "xₖ₊₁ = ΠC[xₖ − η(∇f(xₖ) + μ∇V(xₖ))]",
      explanation: "A projected gradient step includes a penalty V for constraint violation. A sampled feasible check may refine the endpoint.",
      parameter: "η and μ are internal to this approximate reference path.",
    };
    if (family === "socp") return {
      formula: "xₖ₊₁ = ΠC[xₖ − η(∇f(xₖ) + μ∇B(xₖ))]",
      explanation: "A norm-bound direction is followed by a feasible projection. A sampled feasible check may refine the endpoint.",
      parameter: "η and μ are internal to this approximate reference path.",
    };
    if (family === "sdp") return {
      formula: "Xₖ₊₁ ≈ ΠPSD(Xₖ − ηC)",
      explanation: "The trace-one matrix slice is updated and clipped so its eigenvalues stay nonnegative. A sampled feasible check may refine the endpoint.",
      parameter: "η is internal to this approximate reference path.",
    };
    return {
      formula: "xₖ₊₁ ≈ ΠC(xₖ − η dₖ)",
      explanation: "Projected updates keep the path feasible for this family. A sampled feasible check can refine the endpoint.",
      parameter: "The projection and internal direction dₖ depend on the family; this is approximate.",
    };
  }
  if (method === "gd") return {
    formula: "xₖ₊₁ = xₖ − α∇f(xₖ)",
    explanation: "Follow the negative gradient. The step may leave a constrained region and, on a nonconvex surface, may settle in a local valley.",
    parameter: "α is the step size; the iteration cap limits how long the trace runs.",
  };
  if (method === "sgd") return {
    formula: "xₖ₊₁ = xₖ − α(∇f(xₖ) + εₖ)",
    explanation: "Add a small seeded Gaussian perturbation to each gradient. Compare paths with different seeds to see the effect of noise.",
    parameter: "α controls step size; the seed reproduces the same noise sequence.",
  };
  if (method === "pgd") return {
    formula: "xₖ₊₁ = ΠC(xₖ − α∇f(xₖ))",
    explanation: "Take a gradient step, then project back to the feasible set C. Compare it with ordinary gradient descent on the same example.",
    parameter: "α is the step size; projection enforces feasibility after each step.",
  };
  return {
    formula: "xₖ₊₁ = xₖ − α H(xₖ)⁻¹∇f(xₖ)",
    explanation: family === "nonconvex"
      ? "The Hessian H describes local curvature. If it is indefinite, a Newton step can move uphill or toward a saddle."
      : "The Hessian H rescales the gradient by local curvature, often crossing a quadratic bowl in fewer steps.",
    parameter: "α damps the Newton step; the iteration cap limits the trace.",
  };
}
