import { useEffect, useMemo, useRef, useState } from "react";

import { getMethods, getProblems, solveProblem } from "./api";
import { MethodSummary as MethodSummaryPanel } from "./components/MethodSummary";
import { ProblemPlot } from "./components/ProblemPlot";
import type { GeometryResponse, MethodId, MethodSummary, ProblemSummary, ProblemType, SolveResponse } from "./types";

const PROBLEM_TYPES: Array<{ id: ProblemType; label: string; fullName: string; description: string }> = [
  { id: "lp", label: "LP", fullName: "Linear programs", description: "Straight boundaries and a linear objective" },
  { id: "qp", label: "QP", fullName: "Quadratic programs", description: "A curved objective with linear limits" },
  { id: "qcqp", label: "QCQP", fullName: "Quadratically constrained", description: "Curved objectives and curved limits" },
  { id: "socp", label: "SOCP", fullName: "Second-order cone", description: "Norm constraints and cone sections" },
  { id: "sdp", label: "SDP", fullName: "Semidefinite programs", description: "A two-dimensional slice of PSD matrices" },
];

const DEFAULT_METHOD: MethodId = "simplex";

function shortName(problem: ProblemSummary) {
  return problem.name.replace(/^[A-Z]+:\s*/, "");
}

export default function App() {
  const [problems, setProblems] = useState<ProblemSummary[]>([]);
  const [methods, setMethods] = useState<MethodSummary[]>([]);
  const [selectedProblemId, setSelectedProblemId] = useState("lp-basic");
  const [selectedMethod, setSelectedMethod] = useState<MethodId>(DEFAULT_METHOD);
  const [maxIterations, setMaxIterations] = useState(60);
  const [stepSize, setStepSize] = useState(0.12);
  const [seed, setSeed] = useState(7);
  const [settingsDirty, setSettingsDirty] = useState(false);
  const [geometry, setGeometry] = useState<GeometryResponse | null>(null);
  const [result, setResult] = useState<SolveResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const requestId = useRef(0);

  useEffect(() => {
    let cancelled = false;
    Promise.all([getProblems(), getMethods()])
      .then(([problemList, methodList]) => {
        if (cancelled) return;
        setProblems(problemList);
        setMethods(methodList);
        setSelectedProblemId(problemList[0]?.id ?? "lp-basic");
      })
      .catch((caught: Error) => {
        if (!cancelled) setError(caught.message);
      });
    return () => { cancelled = true; };
  }, []);

  const selectedProblem = useMemo(
    () => problems.find((problem) => problem.id === selectedProblemId) ?? null,
    [problems, selectedProblemId],
  );
  const selectedType = selectedProblem?.type ?? "lp";
  const typeInfo = PROBLEM_TYPES.find((type) => type.id === selectedType) ?? PROBLEM_TYPES[0];
  const examples = problems.filter((problem) => problem.type === selectedType);
  const compatibleMethods = methods.filter((method) => selectedProblem?.compatible_methods.includes(method.id));
  const selectedMethodSummary = compatibleMethods.find((method) => method.id === selectedMethod);
  const hasSettings = ["gd", "sgd", "pgd", "newton"].includes(selectedMethod);
  const settingsError = !hasSettings ? null
    : !Number.isInteger(maxIterations) || maxIterations < 1 || maxIterations > 500 ? "Iterations must be between 1 and 500."
    : !Number.isFinite(stepSize) || stepSize <= 0 || stepSize > 2 ? "Step size must be greater than 0 and at most 2."
    : !Number.isInteger(seed) || seed < 0 ? "Random seed must be a nonnegative whole number."
    : null;

  async function runSolve(problemId: string, method: MethodId) {
    const currentRequest = ++requestId.current;
    const usesSettings = ["gd", "sgd", "pgd", "newton"].includes(method);
    if (usesSettings && settingsError) {
      setError(settingsError);
      setLoading(false);
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    setGeometry((current) => current?.problem_id === problemId ? current : null);
    try {
      const response = await solveProblem({
        problem_id: problemId,
        method,
        max_iterations: usesSettings ? maxIterations : 60,
        step_size: usesSettings ? stepSize : 0.12,
        seed: usesSettings ? seed : 7,
      });
      if (currentRequest !== requestId.current) return;
      setResult(response);
      setGeometry(response.geometry);
      setSettingsDirty(false);
    } catch (caught) {
      if (currentRequest !== requestId.current) return;
      setError(caught instanceof Error ? caught.message : "The solve could not be completed.");
    } finally {
      if (currentRequest === requestId.current) setLoading(false);
    }
  }

  useEffect(() => {
    if (!selectedProblem) return;
    if (!selectedProblem.compatible_methods.includes(selectedMethod)) {
      setSelectedMethod(selectedProblem.compatible_methods[0]);
      return;
    }
    void runSolve(selectedProblem.id, selectedMethod);
    return () => { requestId.current += 1; };
    // Parameter edits are applied by the Run button; selecting a new problem or method runs automatically.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedProblem, selectedMethod]);

  function selectType(type: ProblemType) {
    const first = problems.find((problem) => problem.type === type);
    if (first) selectProblem(first.id);
  }

  function selectProblem(problemId: string) {
    if (problemId === selectedProblemId) return;
    requestId.current += 1;
    setResult(null);
    setGeometry(null);
    setLoading(true);
    setSelectedProblemId(problemId);
  }

  function selectMethod(method: MethodId) {
    if (method === selectedMethod) return;
    requestId.current += 1;
    setResult(null);
    setLoading(true);
    setSelectedMethod(method);
  }

  function updateSetting(setter: (value: number) => void, value: number) {
    setter(value);
    setSettingsDirty(true);
  }

  return (
    <main className="app-shell">
      <header className="site-header">
        <div className="brand-mark" aria-hidden="true"><span /><span /><span /><span /></div>
        <div className="brand-copy"><strong>OPTIMIZATION LAB</strong><span>Interactive convex optimization</span></div>
        <span className="header-count">25 guided examples</span>
      </header>

      <section className="hero">
        <div>
          <p className="eyebrow">LEARN BY EXPERIMENTING</p>
          <h1>See how an optimizer <em>finds its way.</em></h1>
          <p className="hero-copy">Explore feasible regions, objective surfaces, and the steps a method takes toward a solution. Change the problem or method to see a new path.</p>
        </div>
        <div className="hero-key" aria-label="Visualization guide">
          <span><i className="key-dot key-feasible" />Feasible region</span>
          <span><i className="key-dot key-path" />Method path</span>
          <span><i className="key-dot key-boundary" />Constraint edge</span>
        </div>
      </section>

      <section className="workspace">
        <aside className="control-panel card" aria-label="Experiment controls">
          <div className="panel-heading"><span className="section-number">01</span><div><h2>Set up an experiment</h2><p>Choose a problem and method.</p></div></div>

          <div className="control-section">
            <h3>Problem family</h3>
            <div className="type-tabs" role="group" aria-label="Problem family">
              {PROBLEM_TYPES.map((type) => (
                <button key={type.id} type="button" className={selectedType === type.id ? "type-tab active" : "type-tab"} aria-pressed={selectedType === type.id} title={type.fullName} onClick={() => selectType(type.id)}>{type.label}</button>
              ))}
            </div>
            <p className="family-caption"><strong>{typeInfo.fullName}</strong> · {typeInfo.description}</p>
          </div>

          <div className="control-section">
            <h3>Example</h3>
            <div className="example-list" role="group" aria-label="Examples">
              {examples.map((problem, index) => (
                <button key={problem.id} type="button" className={problem.id === selectedProblemId ? "example-option active" : "example-option"} aria-pressed={problem.id === selectedProblemId} onClick={() => selectProblem(problem.id)}>
                  <span className="example-index">{String(index + 1).padStart(2, "0")}</span>
                  <span>{shortName(problem)}</span>
                </button>
              ))}
            </div>
            <select className="mobile-example-select" aria-label="Example" value={selectedProblemId} onChange={(event) => selectProblem(event.target.value)}>
              {examples.map((problem) => <option key={problem.id} value={problem.id}>{shortName(problem)}</option>)}
            </select>
          </div>

          <div className="control-section method-section">
            <label htmlFor="method-select">Method</label>
            <select id="method-select" value={selectedMethod} onChange={(event) => selectMethod(event.target.value as MethodId)}>
              {compatibleMethods.map((method) => <option key={method.id} value={method.id}>{method.name}</option>)}
            </select>
            <p className="method-help">{selectedMethodSummary?.description ?? "Methods appear when a problem is selected."}</p>
          </div>

          {hasSettings && (
            <details className="settings-panel">
              <summary>Method settings {settingsDirty && <span className="dirty-indicator">edited</span>}</summary>
              <div className="settings-grid">
                <label>Iterations<input min={1} max={500} type="number" value={maxIterations} onChange={(event) => updateSetting(setMaxIterations, Number(event.target.value))} /></label>
                <label>Step size<input min={0.001} max={2} step={0.01} type="number" value={stepSize} onChange={(event) => updateSetting(setStepSize, Number(event.target.value))} /></label>
                <label>Random seed<input min={0} type="number" value={seed} onChange={(event) => updateSetting(setSeed, Number(event.target.value))} /></label>
              </div>
            </details>
          )}
          {settingsError && <p className="settings-error" role="alert">{settingsError}</p>}

          <button className="run-button" type="button" disabled={!selectedProblem || loading || Boolean(settingsError)} onClick={() => selectedProblem && void runSolve(selectedProblem.id, selectedMethod)}>
            {loading ? "Computing path…" : settingsDirty ? "Apply settings & run" : "Run this method"}<span aria-hidden="true">↗</span>
          </button>
          <p className="run-hint">Selecting an example or method runs it automatically.</p>
          {error && <div className="error-box" role="alert">{error}</div>}
        </aside>

        <div className="main-column">
          {selectedProblem && (
            <section className="problem-overview card" aria-labelledby="problem-title">
              <div className="overview-top"><span className="class-badge">{selectedProblem.type.toUpperCase()}</span><span className="overview-kicker">THE PROBLEM</span></div>
              <h2 id="problem-title">{shortName(selectedProblem)}</h2>
              <p className="problem-description">{selectedProblem.description}</p>
              <div className="problem-facts">
                <div><span className="fact-label">OBJECTIVE</span><strong className="formula">{selectedProblem.objective_expression}</strong></div>
                <div><span className="fact-label">WHAT TO NOTICE</span><strong>{selectedProblem.learning_goal}</strong></div>
              </div>
              <div className="constraints"><span className="fact-label">SUBJECT TO</span><div>{selectedProblem.constraints.map((constraint) => <code key={constraint}>{constraint}</code>)}</div></div>
            </section>
          )}

          <section className="visual-card card" aria-label="Optimization visualization">
            <ProblemPlot geometry={geometry} result={result} loading={loading} />
          </section>

          <MethodSummaryPanel result={result} loading={loading} />
        </div>
      </section>
    </main>
  );
}
