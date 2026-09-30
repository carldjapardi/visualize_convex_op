import { useEffect, useLayoutEffect, useRef, useState, type ComponentType, type CSSProperties } from "react";
import type { Data, Layout } from "plotly.js";

import type { GeometryResponse, Iterate, SolveResponse } from "../types";

type PlotProps = {
  className?: string;
  data: Data[];
  layout: Partial<Layout>;
  config?: object;
  style?: CSSProperties;
  useResizeHandler?: boolean;
};
type PlotComponent = ComponentType<PlotProps>;
type PlotView = "contours" | "value-map" | "surface";

const VIEWS: Array<{ id: PlotView; label: string }> = [
  { id: "contours", label: "2D contours" },
  { id: "value-map", label: "Value map" },
  { id: "surface", label: "3D surface" },
];
const INK = "#203330";
const PATH = "#ec744d";
const EDGE = "#bc6044";

export function ProblemPlot({
  geometry,
  result,
  loading,
}: {
  geometry: GeometryResponse | null;
  result: SolveResponse | null;
  loading: boolean;
}) {
  const { Plot, error } = usePlotlyComponent();
  const [view, setView] = useState<PlotView>("contours");
  const [activeIndex, setActiveIndex] = useState(0);
  const [compact, setCompact] = useState(() => window.matchMedia("(max-width: 700px)").matches);
  const inspectorRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    const query = window.matchMedia("(max-width: 700px)");
    const update = () => setCompact(query.matches);
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);

  useLayoutEffect(() => {
    setActiveIndex(Math.max(0, (result?.trace.length ?? 1) - 1));
  }, [result]);

  const trace = result?.trace ?? [];
  const index = Math.min(activeIndex, Math.max(0, trace.length - 1));
  const visibleTrace = trace.slice(0, index + 1);
  const current = trace[index];
  const data = geometry
    ? view === "surface"
      ? buildSurfaceTraces(geometry, visibleTrace, compact)
      : buildPlanarTraces(geometry, visibleTrace, view)
    : [];
  const layout = geometry
    ? view === "surface" ? surfaceLayout(geometry, compact) : planarLayout(geometry, view, compact)
    : {};

  return (
    <div className="plot-stack">
      <div className="plot-toolbar">
        <div>
          <span className="section-number">02</span>
          <div><h2>Explore the geometry</h2><p>{viewDescription(view)}</p></div>
        </div>
        <div className="view-switch" role="group" aria-label="Visualization view">
          {VIEWS.map((option) => (
            <button
              key={option.id}
              type="button"
              className={view === option.id ? "view-button active" : "view-button"}
              aria-pressed={view === option.id}
              onClick={() => setView(option.id)}
            >{option.label}</button>
          ))}
        </div>
      </div>

      {error ? (
        <div className="plot-empty" role="alert">The plot renderer could not load: {error}</div>
      ) : !geometry || !Plot ? (
        <div className="plot-empty">{loading ? "Computing the objective and feasible region…" : "Loading visualization…"}</div>
      ) : (
        <div className={view === "surface" ? "plot-body surface-layout" : "plot-body"}>
          <div className="plot-frame">
            <Plot
              className="plot"
              data={data}
              layout={layout}
              useResizeHandler
              style={{ width: "100%", height: "100%" }}
              config={{ responsive: true, displayModeBar: true, displaylogo: false, scrollZoom: false }}
            />
          </div>
          <button className="scroll-to-inspector" type="button" onClick={() => inspectorRef.current?.scrollIntoView({ behavior: "smooth", block: "start" })}>Continue to path inspector ↓</button>
          <aside className="trace-inspector" aria-label="Iteration inspector" ref={inspectorRef}>
            <h3>Inspect the path</h3>
            <p className="inspector-intro">Slide through the steps to see where the method moved.</p>
            {current ? (
              <>
                <div className="iteration-header"><span>Iteration</span><strong>{current.iteration} <small>/ {trace.length - 1}</small></strong></div>
                <input
                  className="iteration-range"
                  type="range"
                  min={0}
                  max={trace.length - 1}
                  value={index}
                  aria-label="Iteration"
                  onChange={(event) => setActiveIndex(Number(event.target.value))}
                />
                <div className="iteration-actions">
                  <button type="button" onClick={() => setActiveIndex(0)} disabled={index === 0}>Start</button>
                  <button type="button" onClick={() => setActiveIndex(trace.length - 1)} disabled={index === trace.length - 1}>Final step</button>
                </div>
                <dl className="point-metrics">
                  <div><dt>Objective</dt><dd>{formatNumber(current.objective)}</dd></div>
                  <div><dt>Violation</dt><dd className={current.violation > 1e-5 ? "metric-alert" : ""}>{formatNumber(current.violation)}</dd></div>
                  <div><dt>{geometry.problem_type === "sdp" ? "a, b" : "x, y"}</dt><dd>{current.x.map(formatNumber).join(", ")}</dd></div>
                </dl>
                {current.violation > 1e-5 && <p className="violation-note">This step lies outside the feasible region.</p>}
              </>
            ) : (
              <div className="inspector-empty">Run a method to reveal its path. The colored region contains points that meet every constraint.</div>
            )}
            <div className="inspector-legend">
              <span><i className={view === "surface" ? "legend-swatch feasible-surface" : "legend-swatch feasible"} />Feasible</span>
              <span><i className="legend-swatch path" />Method path</span>
              {view !== "surface" && <span><i className="legend-swatch boundary" />Boundary</span>}
            </div>
          </aside>
        </div>
      )}

      {geometry?.annotations.length ? <div className="annotation-list">{geometry.annotations.map((note) => <span key={note}>{note}</span>)}</div> : null}
      {result && Plot && <HistoryPlot Plot={Plot} result={result} activeIteration={current?.iteration ?? 0} />}
    </div>
  );
}

function buildPlanarTraces(geometry: GeometryResponse, path: Iterate[], view: PlotView): Data[] {
  const traces: Data[] = view === "value-map"
    ? [
        {
          type: "heatmap", x: geometry.x, y: geometry.y, z: geometry.objective,
          colorscale: "Viridis", colorbar: { title: { text: "objective" } },
          hovertemplate: "x=%{x}<br>y=%{y}<br>f=%{z:.3f}<extra></extra>",
        } as Data,
        feasibleRegionTrace(geometry, 0.25),
      ]
    : [
        feasibleRegionTrace(geometry, 0.27),
        {
          type: "contour", x: geometry.x, y: geometry.y, z: geometry.objective,
          contours: { coloring: "lines", showlabels: true, labelfont: { color: INK, size: 11 } },
          colorscale: [[0, "#159985"], [0.5, "#3b79a9"], [1, "#7953a6"]],
          line: { width: 2 }, showscale: false,
          hovertemplate: "f=%{z:.3f}<extra>Objective contour</extra>",
        } as Data,
      ];
  traces.push(...boundaryTraces(geometry));

  if (path.length) {
    traces.push({
      type: "scatter", mode: "lines+markers",
      x: path.map((point) => point.x[0]), y: path.map((point) => point.x[1]),
      line: { color: PATH, width: 3.5 },
      marker: { color: PATH, size: 6, line: { color: "#fff", width: 1 } },
      name: "Method path",
      text: path.map((point) => "iteration " + point.iteration + "<br>f=" + formatNumber(point.objective)),
      hoverinfo: "text",
    } as Data);
    const current = path[path.length - 1];
    traces.push({
      type: "scatter", mode: "markers",
      x: [current.x[0]], y: [current.x[1]],
      marker: { color: PATH, size: 15, line: { color: "#fff", width: 3 } },
      name: "Selected step",
      hovertemplate: "iteration " + current.iteration + "<br>f=" + formatNumber(current.objective) + "<extra></extra>",
    } as Data);
  }
  return traces;
}

function buildSurfaceTraces(geometry: GeometryResponse, path: Iterate[], compact: boolean): Data[] {
  const [low, high] = feasibleRange(geometry);
  const pathLift = Math.max(0.08, (high - low) * 0.025);
  const feasibleZ = geometry.objective.map((row, y) =>
    row.map((value, x) => geometry.feasible[y]?.[x] ? value : null),
  );
  const traces: Data[] = [
    {
      type: "surface", x: geometry.x, y: geometry.y, z: geometry.objective,
      colorscale: [[0, "#d9e0df"], [1, "#aab8b5"]], opacity: 0.33,
      showscale: false, hoverinfo: "skip", name: "Full objective",
    } as Data,
    {
      type: "surface", x: geometry.x, y: geometry.y, z: feasibleZ,
      colorscale: "Viridis", opacity: 0.88, showscale: !compact,
      colorbar: { title: { text: "objective" }, tickfont: { color: INK } },
      hovertemplate: "x=%{x}<br>y=%{y}<br>f=%{z:.3f}<extra>Feasible surface</extra>",
      name: "Feasible objective",
    } as Data,
  ];
  if (path.length) {
    traces.push({
      type: "scatter3d", mode: "lines+markers",
      x: path.map((point) => point.x[0]),
      y: path.map((point) => point.x[1]),
      z: path.map((point) => point.objective + pathLift),
      line: { color: PATH, width: 6 },
      marker: { color: PATH, size: 5 },
      name: "Method path",
      text: path.map((point) => "iteration " + point.iteration + "<br>f=" + formatNumber(point.objective)),
      hoverinfo: "text",
    } as Data);
    const current = path[path.length - 1];
    traces.push({
      type: "scatter3d", mode: "markers",
      x: [current.x[0]], y: [current.x[1]], z: [current.objective + pathLift],
      marker: { color: PATH, size: 10, line: { color: "#fff", width: 2 } },
      name: "Selected step", hoverinfo: "skip",
    } as Data);
  }
  return traces;
}

function feasibleRegionTrace(geometry: GeometryResponse, opacity: number): Data {
  return {
    type: "heatmap", x: geometry.x, y: geometry.y,
    z: geometry.feasible.map((row) => row.map((value) => value ? 1 : null)),
    colorscale: [[0, "#46cfab"], [1, "#46cfab"]], opacity,
    showscale: false, hoverinfo: "skip", name: "Feasible region",
  } as Data;
}

function boundaryTraces(geometry: GeometryResponse): Data[] {
  return geometry.boundaries.map((boundary) => ({
    type: "scatter", mode: "lines", x: boundary.x, y: boundary.y,
    line: { width: 2.5, color: EDGE, dash: "dash" },
    name: boundary.name, hoverinfo: "name",
  } as Data));
}

function planarLayout(geometry: GeometryResponse, view: PlotView, compact: boolean): Partial<Layout> {
  return {
    autosize: true, height: compact ? 430 : 520, uirevision: geometry.problem_id,
    margin: { l: compact ? 48 : 64, r: view === "value-map" ? (compact ? 50 : 72) : 22, t: 20, b: compact ? 44 : 56 },
    paper_bgcolor: "#fff", plot_bgcolor: "#fff",
    font: { color: INK, family: "Inter, system-ui, sans-serif" },
    showlegend: false,
    xaxis: { ...axis(geometry.problem_type === "sdp" ? "a = X₀₀" : "x"), range: [geometry.x[0], geometry.x[geometry.x.length - 1]] },
    yaxis: { ...axis(geometry.problem_type === "sdp" ? "b = X₀₁" : "y"), range: [geometry.y[0], geometry.y[geometry.y.length - 1]], scaleanchor: "x", scaleratio: 1 },
  };
}

function surfaceLayout(geometry: GeometryResponse, compact: boolean): Partial<Layout> {
  const [low, high] = feasibleRange(geometry);
  const margin = Math.max(0.5, (high - low) * 0.16);
  return {
    autosize: true, height: compact ? 460 : 610, uirevision: geometry.problem_id,
    margin: { l: 0, r: 0, t: 8, b: 0 },
    paper_bgcolor: "#fff", plot_bgcolor: "#fff",
    font: { color: INK, family: "Inter, system-ui, sans-serif" },
    showlegend: false,
    scene: {
      bgcolor: "#fff", aspectmode: "manual", aspectratio: { x: 1, y: 1, z: 0.68 },
      xaxis: sceneAxis(geometry.problem_type === "sdp" ? "a = X₀₀" : "x"),
      yaxis: sceneAxis(geometry.problem_type === "sdp" ? "b = X₀₁" : "y"),
      zaxis: { ...sceneAxis("objective"), range: [low - margin, high + margin] },
      camera: { eye: compact ? { x: 1.45, y: 1.55, z: 1.1 } : { x: 1.2, y: 1.35, z: 0.9 } },
    },
  };
}

function feasibleRange(geometry: GeometryResponse): [number, number] {
  const values = geometry.objective.flatMap((row, y) =>
    row.filter((value, x): value is number => Boolean(geometry.feasible[y]?.[x]) && value !== null),
  );
  return values.length ? [Math.min(...values), Math.max(...values)] : [0, 1];
}

function axis(title: string) {
  return {
    title: { text: title }, gridcolor: "#e7eeea",
    linecolor: "#a8b8b1", zerolinecolor: "#cbd8d2",
  };
}

function sceneAxis(title: string) {
  return {
    title: { text: title }, color: INK,
    gridcolor: "#e7eeea", linecolor: "#a8b8b1",
    zerolinecolor: "#cbd8d2",
  };
}

function viewDescription(view: PlotView) {
  if (view === "surface") return "Drag to rotate. Height is objective value (lower is better); the orange path is raised for visibility.";
  if (view === "value-map") return "Color shows objective value; the green overlay satisfies every constraint.";
  return "Curves connect equal objective values. The green area satisfies every constraint.";
}

function HistoryPlot({
  Plot, result, activeIteration,
}: {
  Plot: PlotComponent; result: SolveResponse; activeIteration: number;
}) {
  const iterations = result.trace.map((point) => point.iteration);
  const hasViolation = result.trace.some((point) => point.violation > 1e-5);
  const data: Data[] = [{
    type: "scatter", mode: "lines", x: iterations,
    y: result.trace.map((point) => point.objective),
    line: { color: "#137e76", width: 3 }, name: "Objective",
  } as Data];
  if (hasViolation) {
    data.push({
      type: "scatter", mode: "lines", x: iterations,
      y: result.trace.map((point) => point.violation),
      yaxis: "y2", line: { color: PATH, width: 2 }, name: "Constraint violation",
    } as Data);
  }
  return (
    <div className="history-section">
      <div className="history-heading"><h3>Across the run</h3><p>Objective value{hasViolation ? " and constraint violation" : ""} at each iteration.</p></div>
      <Plot
        className="history-plot" data={data} useResizeHandler
        style={{ width: "100%", height: "100%" }}
        layout={{
          autosize: true, height: 220, uirevision: "history",
          margin: { l: 62, r: hasViolation ? 62 : 20, t: 14, b: 44 },
          paper_bgcolor: "#fff", plot_bgcolor: "#fff",
          font: { color: INK, family: "Inter, system-ui, sans-serif" },
          xaxis: axis("iteration"), yaxis: axis("objective"),
          yaxis2: { ...axis("violation"), overlaying: "y", side: "right", showgrid: false },
          legend: { orientation: "h", y: 1.26 },
          shapes: [{
            type: "line", x0: activeIteration, x1: activeIteration, y0: 0, y1: 1,
            xref: "x", yref: "paper", line: { color: PATH, width: 2, dash: "dot" },
          }],
        }}
        config={{ responsive: true, displayModeBar: false }}
      />
    </div>
  );
}

function formatNumber(value: number) {
  if (Math.abs(value) < 0.00005) return "0";
  if (Math.abs(value) < 0.001 || Math.abs(value) >= 10000) return value.toExponential(2);
  return value.toLocaleString(undefined, { maximumFractionDigits: 4 });
}

function usePlotlyComponent() {
  const [Plot, setPlot] = useState<PlotComponent | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    Promise.all([import("plotly.js-dist-min"), import("react-plotly.js/factory")])
      .then(([plotlyModule, factoryModule]) => {
        if (cancelled) return;
        const factoryDefault = factoryModule.default as unknown as
          | ((plotly: unknown) => PlotComponent)
          | { default: (plotly: unknown) => PlotComponent };
        const create = typeof factoryDefault === "function" ? factoryDefault : factoryDefault.default;
        const plotly = (plotlyModule as { default?: unknown }).default ?? plotlyModule;
        setPlot(() => create(plotly));
      })
      .catch((caught: Error) => { if (!cancelled) setError(caught.message); });
    return () => { cancelled = true; };
  }, []);

  return { Plot, error };
}
