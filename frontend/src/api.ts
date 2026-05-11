import type { GeometryResponse, MethodSummary, ProblemSummary, SolveRequest, SolveResponse } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed with ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export function getProblems(): Promise<ProblemSummary[]> {
  return request<ProblemSummary[]>("/problem-types");
}

export function getMethods(): Promise<MethodSummary[]> {
  return request<MethodSummary[]>("/methods");
}

export function solveProblem(payload: SolveRequest): Promise<SolveResponse> {
  return request<SolveResponse>("/solve", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function sampleGeometry(problemId: string, resolution = 80): Promise<GeometryResponse> {
  return request<GeometryResponse>("/sample-geometry", {
    method: "POST",
    body: JSON.stringify({ problem_id: problemId, resolution }),
  });
}
