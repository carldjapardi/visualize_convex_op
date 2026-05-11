import { useEffect, useMemo, useState } from "react";

import { getMethods, getProblems, sampleGeometry, solveProblem } from "./api";
import { MethodSummary as MethodSummaryPanel } from "./components/MethodSummary";
import { ProblemPlot } from "./components/ProblemPlot";
import type { GeometryResponse, MethodId, MethodSummary, ProblemSummary, SolveResponse } from "./types";

const DEFAULT_METHOD: MethodId = "interior_point";

export default function App() {
  const [problems, setProblems] = useState<ProblemSummary[]>([]);
  const [methods, setMethods] = useState<MethodSummary[]>([]);
  const [selectedProblemId, setSelectedProblemId] = useState("lp-basic");
  const [selectedMethod, setSelectedMethod] = useState<MethodId>(DEFAULT_METHOD);
  const [maxIterations, setMaxIterations] = useState(60);
  const [stepSize, setStepSize] = useState(0.12);
  const [seed, setSeed] = useState(7);
  const [geometry, setGeometry] = useState<GeometryResponse | null>(null);
  const [result, setResult] = useState<SolveResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let ignore = false;

    Promise.all([getProblems(), getMethods()])
      .then(([problemList, methodList]) => {
        if (ignore) {
          return;
        }
        setProblems(problemList);
        setMethods(methodList);
        setSelectedProblemId(problemList[0]?.id ?? "lp-basic");
      })
      .catch((caught: Error) => setError(caught.message));

    return () => {
      ignore = true;
    };
  }, []);

  const selectedProblem = useMemo(
    () => problems.find((problem) => problem.id === selectedProblemId) ?? null,
    [problems, selectedProblemId],
  );

  const selectedMethodSummary = useMemo(
    () => methods.find((method) => method.id === selectedMethod) ?? null,
    [methods, selectedMethod],
  );

  const isTraceMethod = selectedMethod === "gd" || selectedMethod === "sgd" || selectedMethod === "pgd" || selectedMethod === "newton";

  useEffect(() => {
    if (!selectedProblem) {
      return;
    }

    setError(null);
    setResult(null);
    sampleGeometry(selectedProblem.id)
      .then(setGeometry)
      .catch((caught: Error) => setError(caught.message));
  }, [selectedProblem]);

  useEffect(() => {
    if (!selectedProblem) {
      return;
    }
    if (!selectedProblem.compatible_methods.includes(selectedMethod)) {
      setSelectedMethod(selectedProblem.compatible_methods[0] ?? DEFAULT_METHOD);
    }
  }, [selectedProblem, selectedMethod]);

  async function handleSolve() {
    if (!selectedProblem) {
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const response = await solveProblem({
        problem_id: selectedProblem.id,
        method: selectedMethod,
        max_iterations: maxIterations,
        step_size: stepSize,
        seed,
      });
      setResult(response);
      if (response.geometry) {
        setGeometry(response.geometry);
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Convex Optimization Lab</p>
          <h1>Visualize LPs, QPs, QCQPs, SOCPs, and SDPs</h1>
          <p>
            Compare real solver outputs with method traces, feasible regions, objective contours, and matrix-cone diagnostics.
          </p>
        </div>
      </header>

      <section className="workspace">
        <aside className="control-panel card">
          <h2>Experiment</h2>

          <label>
            Problem
            <select value={selectedProblemId} onChange={(event) => setSelectedProblemId(event.target.value)}>
              {problems.map((problem) => (
                <option key={problem.id} value={problem.id}>
                  {problem.type.toUpperCase()} - {problem.name}
                </option>
              ))}
            </select>
          </label>

          <label>
            Method
            <select value={selectedMethod} onChange={(event) => setSelectedMethod(event.target.value as MethodId)}>
              {methods.map((method) => (
                <option key={method.id} value={method.id}>
                  {method.name}
                </option>
              ))}
            </select>
          </label>

          {selectedProblem && selectedMethodSummary && (
            <div className="compatibility">
              {selectedProblem.compatible_methods.includes(selectedMethod)
                ? `${selectedMethodSummary.name} is enabled for this ${selectedProblem.type.toUpperCase()} example.`
                : `${selectedMethodSummary.name} is not mathematically appropriate for this example.`}
            </div>
          )}

          {isTraceMethod && (
            <div className="parameter-grid">
              <label>
                Iterations
                <input
                  min={1}
                  max={500}
                  type="number"
                  value={maxIterations}
                  onChange={(event) => setMaxIterations(Number(event.target.value))}
                />
              </label>
              <label>
                Step size
                <input
                  min={0.001}
                  max={2}
                  step={0.01}
                  type="number"
                  value={stepSize}
                  onChange={(event) => setStepSize(Number(event.target.value))}
                />
              </label>
              <label>
                Seed
                <input min={0} type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value))} />
              </label>
            </div>
          )}

          <button disabled={!selectedProblem || loading} onClick={handleSolve}>
            {loading ? "Solving..." : "Solve and Visualize"}
          </button>

          {error && <div className="error-box">{error}</div>}

          {selectedProblem && (
            <div className="problem-notes">
              <h3>{selectedProblem.name}</h3>
              <p>{selectedProblem.description}</p>
              <h4>Constraints</h4>
              <ul>
                {selectedProblem.constraints.map((constraint) => (
                  <li key={constraint}>{constraint}</li>
                ))}
              </ul>
            </div>
          )}
        </aside>

        <section className="visual-area">
          <div className="card plot-card">
            <ProblemPlot geometry={geometry} result={result} />
          </div>
          <MethodSummaryPanel result={result} />
        </section>
      </section>
    </main>
  );
}
