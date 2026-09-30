import { useEffect, useMemo, useRef, useState } from "react";

import { getMethods, getProblems, solveProblem } from "./api";
import { MethodSummary as MethodSummaryPanel } from "./components/MethodSummary";
import { ProblemPlot } from "./components/ProblemPlot";
import { TeachingPanel } from "./components/TeachingPanel";
import { FAMILY_GUIDES } from "./teaching";
import type { GeometryResponse, MethodId, MethodSummary, ProblemSummary, ProblemType, SolveResponse } from "./types";

const PROBLEM_TYPES: Array<{ id: ProblemType; label: string }> = [
  { id: "lp", label: "LP" }, { id: "qp", label: "QP" }, { id: "qcqp", label: "QCQP" },
  { id: "socp", label: "SOCP" }, { id: "sdp", label: "SDP" }, { id: "nonconvex", label: "NONCONVEX" },
];

const START_PRESETS: Record<string, Array<{ label: string; point: [number, number] }>> = {
  "nonconvex-double-well": [
    { label: "Left basin", point: [-0.3, 1.15] }, { label: "Right basin", point: [0.3, 1.15] },
  ],
  "nonconvex-himmelblau": [
    { label: "Center", point: [0, 0] }, { label: "Upper left", point: [-3, 3] }, { label: "Lower right", point: [3, -3] },
  ],
  "nonconvex-rippled-bowl": [
    { label: "Outer ripple", point: [2.4, -1.8] }, { label: "Near center", point: [0.4, 0.4] },
  ],
};

const DEFAULT_METHOD: MethodId = "simplex";

function shortName(problem: ProblemSummary) {
  return problem.name.replace(/^[A-Za-z]+:\s*/, "");
}

export default function App() {
  const [problems, setProblems] = useState<ProblemSummary[]>([]);
  const [methods, setMethods] = useState<MethodSummary[]>([]);
  const [selectedProblemId, setSelectedProblemId] = useState("lp-basic");
  const [selectedMethod, setSelectedMethod] = useState<MethodId>(DEFAULT_METHOD);
  const [maxIterations, setMaxIterations] = useState(60);
  const [stepSize, setStepSize] = useState(0.12);
  const [seed, setSeed] = useState(7);
  const [startPoint, setStartPoint] = useState<[number, number]>([0.5, 0.5]);
  const [activeIndex, setActiveIndex] = useState(0);
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
        if (problemList[0]) {
          setMaxIterations(problemList[0].default_max_iterations);
          setStepSize(problemList[0].default_step_size);
          setStartPoint(problemList[0].initial_point as [number, number]);
        }
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
  const familyGuide = FAMILY_GUIDES[selectedType];
  const examples = problems.filter((problem) => problem.type === selectedType);
  const compatibleMethods = methods.filter((method) => selectedProblem?.compatible_methods.includes(method.id));
  const selectedMethodSummary = compatibleMethods.find((method) => method.id === selectedMethod);
  const hasSettings = ["gd", "sgd", "pgd", "newton"].includes(selectedMethod);
  const settingsError = !hasSettings ? null
    : !Number.isInteger(maxIterations) || maxIterations < 1 || maxIterations > 500 ? "Iterations must be between 1 and 500."
    : !Number.isFinite(stepSize) || stepSize <= 0 || stepSize > 2 ? "Step size must be greater than 0 and at most 2."
    : !Number.isInteger(seed) || seed < 0 ? "Random seed must be a nonnegative whole number."
    : selectedType === "nonconvex" && (!startPoint.every(Number.isFinite) || startPoint.some((value, index) => value < selectedProblem!.plot_bounds[index][0] || value > selectedProblem!.plot_bounds[index][1]))
      ? "Keep the starting point inside the plotted window."
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
    setActiveIndex(0);
    setGeometry((current) => current?.problem_id === problemId ? current : null);
    try {
      const response = await solveProblem({
        problem_id: problemId,
        method,
        max_iterations: usesSettings ? maxIterations : 60,
        step_size: usesSettings ? stepSize : 0.12,
        seed: usesSettings ? seed : 7,
        start_point: selectedType === "nonconvex" ? startPoint : undefined,
      });
      if (currentRequest !== requestId.current) return;
      setResult(response);
      setActiveIndex(Math.max(0, response.trace.length - 1));
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
    // Selecting a new problem or method runs immediately; parameter edits use the short debounce below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedProblem, selectedMethod]);

  useEffect(() => {
    if (!settingsDirty || !selectedProblem) return;
    if (settingsError) {
      setLoading(false);
      return;
    }
    const timer = window.setTimeout(() => void runSolve(selectedProblem.id, selectedMethod), 320);
    return () => window.clearTimeout(timer);
    // Parameter changes recompute after a short pause; selection changes run immediately above.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [settingsDirty, maxIterations, stepSize, seed, startPoint, settingsError]);

  function selectType(type: ProblemType) {
    const first = problems.find((problem) => problem.type === type);
    if (first) selectProblem(first.id);
  }

  function selectProblem(problemId: string) {
    if (problemId === selectedProblemId) return;
    const next = problems.find((problem) => problem.id === problemId);
    requestId.current += 1;
    setResult(null);
    setGeometry(null);
    setLoading(true);
    setSettingsDirty(false);
    setActiveIndex(0);
    if (next) {
      setMaxIterations(next.default_max_iterations);
      setStepSize(next.default_step_size);
      setStartPoint(next.initial_point as [number, number]);
    }
    setSelectedProblemId(problemId);
  }

  function selectMethod(method: MethodId) {
    if (method === selectedMethod) return;
    requestId.current += 1;
    setResult(null);
    setLoading(true);
    setSettingsDirty(false);
    setActiveIndex(0);
    setSelectedMethod(method);
  }

  function updateSetting(setter: (value: number) => void, value: number) {
    requestId.current += 1;
    setResult(null);
    setLoading(true);
    setter(value);
    setSettingsDirty(true);
  }

  function updateStartPoint(point: [number, number]) {
    requestId.current += 1;
    setResult(null);
    setLoading(true);
    setStartPoint(point);
    setSettingsDirty(true);
  }

  return (
    <main className="app-shell">
      <header className="site-header">
        <div className="brand-mark" aria-hidden="true"><span /><span /><span /><span /></div>
        <div className="brand-copy"><strong>OPTIMIZATION LAB</strong><span>Interactive optimization</span></div>
        <span className="header-count">{problems.length ? `${problems.length} guided examples` : "Loading examples"}</span>
      </header>

      <section className="hero">
        <div>
          <p className="eyebrow">LEARN BY EXPERIMENTING</p>
          <h1>See how an optimizer <em>finds its way.</em></h1>
          <p className="hero-copy">Explore feasible regions, loss surfaces, and the steps a method takes. Change the method, parameters, or starting point to see a new path.</p>
        </div>
        <div className="hero-key" aria-label="Visualization guide">
          <span><i className="key-dot key-feasible" />{selectedType === "nonconvex" ? "Loss surface" : "Feasible region"}</span>
          <span><i className="key-dot key-path" />Method path</span>
          <span><i className="key-dot key-boundary" />{selectedType === "nonconvex" ? "Local valley" : "Constraint edge"}</span>
        </div>
      </section>

      <section className="workspace">
        <aside className="control-panel card" aria-label="Experiment controls">
          <div className="panel-heading"><span className="section-number">01</span><div><h2>Set up an experiment</h2><p>Choose a problem and method.</p></div></div>

          <div className="control-section">
            <h3>Problem family</h3>
            <div className="type-tabs" role="group" aria-label="Problem family">
              {PROBLEM_TYPES.map((type) => (
                <button key={type.id} type="button" className={selectedType === type.id ? "type-tab active" : "type-tab"} aria-pressed={selectedType === type.id} title={FAMILY_GUIDES[type.id].name} onClick={() => selectType(type.id)}>{type.label}</button>
              ))}
            </div>
            <p className="family-caption"><strong>{familyGuide.name}</strong> · {familyGuide.short}</p>
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
            <div className="settings-panel" aria-label="Method parameters">
              <strong className="parameter-heading">Tune the path {settingsDirty && <span className="dirty-indicator">updating…</span>}</strong>
              <div className="parameter-control">
                <label htmlFor="iteration-input">Iteration cap <span>{maxIterations}</span></label>
                <input className="parameter-range" aria-label="Iteration cap slider" type="range" min={1} max={200} step={1} value={Math.min(maxIterations, 200)} onChange={(event) => updateSetting(setMaxIterations, Number(event.target.value))} />
                <input id="iteration-input" type="number" min={1} max={500} value={maxIterations} onChange={(event) => updateSetting(setMaxIterations, Number(event.target.value))} />
              </div>
              <div className="parameter-control">
                <label htmlFor="step-input">Step size α <span>{stepSize}</span></label>
                <input className="parameter-range" aria-label="Step size slider" type="range" min={0.005} max={0.3} step={0.005} value={Math.min(Math.max(stepSize, 0.005), 0.3)} onChange={(event) => updateSetting(setStepSize, Number(event.target.value))} />
                <input id="step-input" type="number" min={0.001} max={2} step={0.005} value={stepSize} onChange={(event) => updateSetting(setStepSize, Number(event.target.value))} />
              </div>
              {selectedMethod === "sgd" && <label className="seed-control">Random seed<input min={0} type="number" value={seed} onChange={(event) => updateSetting(setSeed, Number(event.target.value))} /></label>}
              {selectedType === "nonconvex" && selectedProblem && (
                <div className="start-controls">
                  <strong>Starting point</strong>
                  <div className="start-inputs">
                    <label>x<input aria-label="Starting x" type="number" step={0.1} min={selectedProblem.plot_bounds[0][0]} max={selectedProblem.plot_bounds[0][1]} value={startPoint[0]} onChange={(event) => updateStartPoint([Number(event.target.value), startPoint[1]])} /></label>
                    <label>y<input aria-label="Starting y" type="number" step={0.1} min={selectedProblem.plot_bounds[1][0]} max={selectedProblem.plot_bounds[1][1]} value={startPoint[1]} onChange={(event) => updateStartPoint([startPoint[0], Number(event.target.value)])} /></label>
                  </div>
                  <div className="start-presets">
                    {(START_PRESETS[selectedProblem.id] ?? []).map((preset) => <button key={preset.label} type="button" onClick={() => updateStartPoint(preset.point)}>{preset.label}</button>)}
                  </div>
                </div>
              )}
            </div>
          )}
          {settingsError && <p className="settings-error" role="alert">{settingsError}</p>}

          <button className="run-button" type="button" disabled={!selectedProblem || loading || Boolean(settingsError)} onClick={() => selectedProblem && void runSolve(selectedProblem.id, selectedMethod)}>
            {loading ? "Computing path…" : "Run this method"}<span aria-hidden="true">↗</span>
          </button>
          <p className="run-hint">Examples, methods, and parameters update the path automatically.</p>
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

          {selectedProblem && (
            <TeachingPanel
              family={selectedType} method={selectedMethod}
              methodName={selectedMethodSummary?.name ?? "Method"}
              stepSize={stepSize} result={result} activeIndex={activeIndex}
            />
          )}

          <section className="visual-card card" aria-label="Optimization visualization">
            <ProblemPlot
              problemId={selectedProblemId} problemType={selectedType}
              geometry={geometry} result={result} loading={loading}
              activeIndex={activeIndex} onIndexChange={setActiveIndex}
            />
          </section>

          <MethodSummaryPanel result={result} loading={loading} />
        </div>
      </section>
    </main>
  );
}
