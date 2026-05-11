export type ProblemType = "lp" | "qp" | "qcqp" | "socp" | "sdp";
export type MethodId = "simplex" | "interior_point" | "gd" | "sgd" | "pgd" | "newton";

export interface ProblemSummary {
  id: string;
  type: ProblemType;
  name: string;
  description: string;
  dimension: number;
  variables: string[];
  compatible_methods: MethodId[];
  constraints: string[];
}

export interface MethodSummary {
  id: MethodId;
  name: string;
  description: string;
  best_for: ProblemType[];
}

export interface Iterate {
  iteration: number;
  x: number[];
  objective: number;
  violation: number;
  step_norm: number;
}

export interface GeometryResponse {
  problem_id: string;
  problem_type: ProblemType;
  x: number[];
  y: number[];
  objective: Array<Array<number | null>>;
  feasible: number[][];
  boundaries: Array<{ name: string; x: number[]; y: number[] }>;
  annotations: string[];
}

export interface SolveResponse {
  problem: ProblemSummary;
  method: MethodSummary;
  status: string;
  compatible: boolean;
  solver: string | null;
  message: string;
  objective_value: number | null;
  variables: Record<string, unknown>;
  iterations: number;
  trace: Iterate[];
  history: Record<string, number[]>;
  geometry: GeometryResponse | null;
}

export interface SolveRequest {
  problem_id: string;
  method: MethodId;
  max_iterations: number;
  step_size: number;
  seed: number;
}
