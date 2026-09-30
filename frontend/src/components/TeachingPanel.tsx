import { FAMILY_GUIDES, methodGuide } from "../teaching";
import type { MethodId, ProblemType, SolveResponse } from "../types";

export function TeachingPanel({
  family, method, methodName, stepSize, result, activeIndex,
}: {
  family: ProblemType;
  method: MethodId;
  methodName: string;
  stepSize: number;
  result: SolveResponse | null;
  activeIndex: number;
}) {
  const familyGuide = FAMILY_GUIDES[family];
  const guide = methodGuide(method, family);
  const trace = result?.trace ?? [];
  const index = Math.min(activeIndex, trace.length - 1);
  const current = trace[index];
  const previous = index > 0 ? trace[index - 1] : null;
  const change = previous && current ? current.objective - previous.objective : null;

  return (
    <section className="teaching-card card" aria-label="Concept and method guide">
      <div className="teaching-column">
        <span className="teaching-label">ABOUT THE FAMILY</span>
        <h3>{familyGuide.name}</h3>
        <code className="teaching-formula">{familyGuide.form}</code>
        <p>{familyGuide.explanation}</p>
        <small>{familyGuide.focus}</small>
      </div>
      <div className="teaching-column">
        <span className="teaching-label">HOW THE METHOD MOVES</span>
        <h3>{methodName}</h3>
        <code className="teaching-formula">{guide.formula}</code>
        <p>{guide.explanation}</p>
        <small>{guide.parameter}</small>
        {current && (
          <div className="step-readout" aria-live="polite">
            <strong>Selected step {current.iteration}</strong>
            <span>{previous && change !== null
              ? `Moved ${formatMagnitude(current.step_norm)}; ${family === "nonconvex" ? "loss" : "objective"} ${change < -1e-6 ? "fell" : change > 1e-6 ? "rose" : "stayed level"} by ${formatMagnitude(Math.abs(change))}.`
              : "This is the starting point."}</span>
            {["gd", "sgd", "pgd", "newton"].includes(method) && <span>Current α = {stepSize}</span>}
          </div>
        )}
      </div>
    </section>
  );
}

function formatMagnitude(value: number) {
  return value > 0 && value < 0.001 ? "<0.001" : value.toFixed(3);
}
