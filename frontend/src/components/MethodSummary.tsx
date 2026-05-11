import type { SolveResponse } from "../types";

export function MethodSummary({ result }: { result: SolveResponse | null }) {
  if (!result) {
    return (
      <section className="card">
        <h2>Solver Summary</h2>
        <p>Run a method to see the optimizer status, objective, variables, and trace diagnostics.</p>
      </section>
    );
  }

  return (
    <section className="card">
      <div className="summary-heading">
        <div>
          <h2>{result.method.name}</h2>
          <p>{result.method.description}</p>
        </div>
        <span className={`status-pill ${result.compatible ? "status-ok" : "status-warn"}`}>{result.status}</span>
      </div>

      <dl className="metric-grid">
        <div>
          <dt>Objective</dt>
          <dd>{formatNumber(result.objective_value)}</dd>
        </div>
        <div>
          <dt>Iterations</dt>
          <dd>{result.iterations}</dd>
        </div>
        <div>
          <dt>Solver</dt>
          <dd>{result.solver ?? "n/a"}</dd>
        </div>
      </dl>

      <p className="message">{result.message}</p>

      <div>
        <h3>Variables</h3>
        <pre className="json-block">{JSON.stringify(result.variables, null, 2)}</pre>
      </div>
    </section>
  );
}

function formatNumber(value: number | null) {
  if (value === null || Number.isNaN(value)) {
    return "n/a";
  }
  return value.toLocaleString(undefined, { maximumFractionDigits: 6 });
}
