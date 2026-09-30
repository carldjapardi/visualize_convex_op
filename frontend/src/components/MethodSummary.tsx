import type { SolveResponse } from "../types";

export function MethodSummary({ result, loading }: { result: SolveResponse | null; loading: boolean }) {
  if (!result) {
    return (
      <section className="result-card card" aria-label="Solver result">
        <div className="result-placeholder">
          <span className="section-number">03</span>
          <div>
            <h2>{loading ? "Following the method…" : "Review the result"}</h2>
            <p>{loading ? "The current path and objective values are being computed." : "Choose an example to see the method's result here."}</p>
          </div>
        </div>
      </section>
    );
  }

  const final = result.trace[result.trace.length - 1];
  const bestIteration = result.variables.best_iteration;
  const isFirstOrder = typeof bestIteration === "number";
  const isNonconvex = result.problem.type === "nonconvex";
  const violation = final?.violation ?? Number.NaN;
  const pathViolates = result.trace.some((point) => point.violation > 1e-5);
  const statusClass = result.status === "optimal" ? "status-good" : result.status === "approximate" ? "status-approximate" : "status-warning";
  const eigenvalues = result.variables.eigenvalues;
  const matrix = result.variables.X;

  return (
    <section className="result-card card" aria-label="Solver result">
      <div className="result-heading">
        <div><span className="section-number">03</span><div><h2>Review the result</h2><p>{result.method.name} · {result.solver ?? "Educational solver"}</p></div></div>
        <span className={"status-pill " + statusClass}>{result.status.replace(/_/g, " ")}</span>
      </div>
      <div className="result-metrics">
        <div><span>{isNonconvex ? "BEST LOSS" : isFirstOrder ? "BEST OBJECTIVE" : "OBJECTIVE"}</span><strong>{formatNumber(result.objective_value)}</strong></div>
        <div><span>STEPS</span><strong>{result.iterations}</strong></div>
        <div><span>{isNonconvex ? "LANDSCAPE" : isFirstOrder ? "PATH" : "FEASIBILITY"}</span><strong>{isNonconvex ? "Local search" : isFirstOrder ? pathViolates ? "Leaves feasible set" : "Feasible" : Number.isNaN(violation) ? "—" : violation <= 1e-5 ? "Satisfied" : "Violated"}</strong></div>
      </div>
      <p className="result-message">{result.message}</p>
      {Array.isArray(eigenvalues) && Array.isArray(matrix) && (
        <div className="matrix-diagnostics">
          <div><span>PSD EIGENVALUES</span><strong>{eigenvalues.map((value) => typeof value === "number" ? value.toFixed(4) : String(value)).join("  ·  ")}</strong></div>
          <div><span>RESULT MATRIX X</span><code>{JSON.stringify(matrix)}</code></div>
        </div>
      )}
      {isFirstOrder && <p className="accuracy-note">Best visited point: iteration {bestIteration}. The path inspector shows the selected step.</p>}
      {isNonconvex && <p className="accuracy-note">This path does not certify a global minimum. Try another starting point to compare basins.</p>}
      {result.status === "approximate" && <p className="accuracy-note">This in-repo reference is an educational approximation, not a certified optimum.</p>}
      <details className="raw-result"><summary>Inspect numerical output</summary><pre>{JSON.stringify(result.variables, null, 2)}</pre></details>
    </section>
  );
}

function formatNumber(value: number | null) {
  if (value === null || Number.isNaN(value)) return "—";
  return value.toLocaleString(undefined, { maximumFractionDigits: 6 });
}
