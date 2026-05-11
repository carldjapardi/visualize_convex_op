import { useEffect, useState, type ComponentType } from "react";
import type { Data, Layout } from "plotly.js";

import type { GeometryResponse, SolveResponse } from "../types";

type PlotProps = {
  className?: string;
  data: Data[];
  layout: Partial<Layout>;
  config?: object;
};

type PlotComponent = ComponentType<PlotProps>;

interface ProblemPlotProps {
  geometry: GeometryResponse | null;
  result: SolveResponse | null;
}

type PlotView = "level-curves" | "actual-graph" | "surface-3d";

const PLOT_VIEW_OPTIONS: Array<{ value: PlotView; label: string }> = [
  { value: "level-curves", label: "2D level curves" },
  { value: "actual-graph", label: "Actual graph" },
  { value: "surface-3d", label: "3D graph" },
];

const PLOT_TEXT_COLOR = "#111827";

export function ProblemPlot({ geometry, result }: ProblemPlotProps) {
  const { Plot, error } = usePlotlyComponent();
  const [plotView, setPlotView] = useState<PlotView>("level-curves");
  const plotGeometry = result?.geometry ?? geometry;

  if (!plotGeometry) {
    return <div className="empty-state">Loading optimization geometry...</div>;
  }

  if (error) {
    return <div className="empty-state">Plot renderer failed to load: {error}</div>;
  }

  if (!Plot) {
    return <div className="empty-state">Loading plot renderer...</div>;
  }

  const traces =
    plotView === "surface-3d"
      ? buildSurfaceTraces(plotGeometry, result)
      : buildPlanarTraces(plotGeometry, result, plotView);
  const layout =
    plotView === "surface-3d"
      ? buildSurfaceLayout(plotGeometry)
      : buildPlanarLayout(plotGeometry, plotView);

  return (
    <div className="plot-stack">
      <div className="plot-toolbar">
        <div>
          <h2>Visualization</h2>
          <p>{viewDescription(plotView)}</p>
        </div>
        <label>
          View
          <select value={plotView} onChange={(event) => setPlotView(event.target.value as PlotView)}>
            {PLOT_VIEW_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
      </div>
      <Plot className="plot" data={traces} layout={layout} config={{ responsive: true, displayModeBar: true }} />
      {plotGeometry.annotations.length > 0 && (
        <div className="annotation-list">
          {plotGeometry.annotations.map((annotation) => (
            <span key={annotation}>{annotation}</span>
          ))}
        </div>
      )}
      {result && <HistoryPlot Plot={Plot} result={result} />}
    </div>
  );
}

function buildPlanarTraces(geometry: GeometryResponse, result: SolveResponse | null, plotView: PlotView): Data[] {
  const traces: Data[] =
    plotView === "actual-graph"
      ? [
          {
            type: "heatmap",
            x: geometry.x,
            y: geometry.y,
            z: geometry.objective,
            colorscale: "Viridis",
            colorbar: {
              title: { text: "objective" },
              tickfont: { color: PLOT_TEXT_COLOR },
              titlefont: { color: PLOT_TEXT_COLOR },
            },
            hovertemplate: "x=%{x}<br>y=%{y}<br>f=%{z}<extra>Objective</extra>",
            name: "Objective value",
          } as Data,
          feasibleRegionTrace(geometry, 0.2),
        ]
      : [
          feasibleRegionTrace(geometry, 0.18),
          {
            type: "contour",
            x: geometry.x,
            y: geometry.y,
            z: geometry.objective,
            colorscale: [
              [0, "#2563eb"],
              [0.5, "#7c3aed"],
              [1, "#be123c"],
            ],
            contours: {
              coloring: "lines",
              showlabels: true,
              labelfont: { color: PLOT_TEXT_COLOR, size: 11 },
            },
            line: { width: 2.4 },
            opacity: 0.98,
            showscale: false,
            name: "Objective contours",
          } as Data,
        ];

  traces.push(...boundaryTraces(geometry));

  if (result?.trace.length) {
    traces.push({
      type: "scatter",
      mode: "lines+markers",
      x: result.trace.map((point) => point.x[0]),
      y: result.trace.map((point) => point.x[1]),
      marker: { size: 9, color: "#ff6b35", line: { color: PLOT_TEXT_COLOR, width: 1.5 } },
      line: { width: 3.5, color: "#ff6b35" },
      name: "Method trace",
      text: result.trace.map(
        (point) =>
          `iter ${point.iteration}<br>objective ${point.objective.toFixed(4)}<br>violation ${point.violation.toExponential(2)}`,
      ),
      hoverinfo: "text",
    });
  }

  return traces;
}

function buildSurfaceTraces(geometry: GeometryResponse, result: SolveResponse | null): Data[] {
  const traces: Data[] = [
    {
      type: "surface",
      x: geometry.x,
      y: geometry.y,
      z: geometry.objective,
      colorscale: "Viridis",
      opacity: 0.94,
      colorbar: {
        title: { text: "objective" },
        tickfont: { color: PLOT_TEXT_COLOR },
        titlefont: { color: PLOT_TEXT_COLOR },
      },
      contours: {
        z: {
          show: true,
          usecolormap: true,
          highlightcolor: "#ffffff",
          project: { z: true },
        },
      },
      name: "Objective surface",
    } as Data,
  ];

  const feasiblePoints = feasibleSurfacePoints(geometry);
  if (feasiblePoints.x.length) {
    traces.push({
      type: "scatter3d",
      mode: "markers",
      x: feasiblePoints.x,
      y: feasiblePoints.y,
      z: feasiblePoints.z,
      marker: { size: 2.6, color: "#42f59e", opacity: 0.8 },
      name: "Feasible samples",
      hoverinfo: "skip",
    } as Data);
  }

  if (result?.trace.length) {
    traces.push({
      type: "scatter3d",
      mode: "lines+markers",
      x: result.trace.map((point) => point.x[0]),
      y: result.trace.map((point) => point.x[1]),
      z: result.trace.map((point) => objectiveAtNearest(geometry, point.x[0], point.x[1])),
      marker: { size: 5, color: "#ff6b35", line: { color: PLOT_TEXT_COLOR, width: 1 } },
      line: { width: 5, color: "#ff6b35" },
      name: "Method trace",
      text: result.trace.map(
        (point) =>
          `iter ${point.iteration}<br>objective ${point.objective.toFixed(4)}<br>violation ${point.violation.toExponential(2)}`,
      ),
      hoverinfo: "text",
    } as Data);
  }

  return traces;
}

function feasibleRegionTrace(geometry: GeometryResponse, opacity: number): Data {
  return {
    type: "heatmap",
    x: geometry.x,
    y: geometry.y,
    z: geometry.feasible.map((row) => row.map((value) => (value ? 1 : null))),
    colorscale: [
      [0, "rgba(55, 214, 139, 0.04)"],
      [1, "rgba(55, 214, 139, 0.95)"],
    ],
    opacity,
    showscale: false,
    hoverinfo: "skip",
    name: "Feasible region",
  } as Data;
}

function boundaryTraces(geometry: GeometryResponse): Data[] {
  return geometry.boundaries.map(
    (boundary): Data =>
      ({
        type: "scatter",
        mode: "lines",
        x: boundary.x,
        y: boundary.y,
        line: { width: 3, color: "#9a3412", dash: "dot" },
        name: boundary.name,
      }) as Data,
  );
}

function buildPlanarLayout(geometry: GeometryResponse, plotView: PlotView): Partial<Layout> {
  return {
    autosize: true,
    height: 560,
    margin: { l: 56, r: plotView === "actual-graph" ? 72 : 28, t: 24, b: 54 },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    font: { color: PLOT_TEXT_COLOR },
    xaxis: planarAxis(axisLabels(geometry).x),
    yaxis: { ...planarAxis(axisLabels(geometry).y), scaleanchor: "x" },
    legend: { orientation: "h", y: -0.18, font: { color: PLOT_TEXT_COLOR } },
  };
}

function buildSurfaceLayout(geometry: GeometryResponse): Partial<Layout> {
  return {
    autosize: true,
    height: 620,
    margin: { l: 0, r: 0, t: 12, b: 0 },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    font: { color: PLOT_TEXT_COLOR },
    legend: { orientation: "h", y: 0, font: { color: PLOT_TEXT_COLOR } },
    scene: {
      bgcolor: "#ffffff",
      xaxis: surfaceAxis(axisLabels(geometry).x),
      yaxis: surfaceAxis(axisLabels(geometry).y),
      zaxis: surfaceAxis("objective"),
      camera: { eye: { x: 1.55, y: 1.55, z: 1.05 } },
    },
  };
}

function planarAxis(title: string) {
  return {
    title: { text: title },
    gridcolor: "rgba(17, 24, 39, 0.12)",
    linecolor: "rgba(17, 24, 39, 0.5)",
    tickcolor: "rgba(17, 24, 39, 0.5)",
    zeroline: true,
    zerolinecolor: "rgba(17, 24, 39, 0.32)",
  };
}

function surfaceAxis(title: string) {
  return {
    title: { text: title },
    color: PLOT_TEXT_COLOR,
    gridcolor: "rgba(17, 24, 39, 0.12)",
    linecolor: "rgba(17, 24, 39, 0.5)",
    zerolinecolor: "rgba(17, 24, 39, 0.32)",
  };
}

function axisLabels(geometry: GeometryResponse) {
  return geometry.problem_type === "sdp" ? { x: "a = X00", y: "b = X01" } : { x: "x", y: "y" };
}

function feasibleSurfacePoints(geometry: GeometryResponse) {
  const x: number[] = [];
  const y: number[] = [];
  const z: number[] = [];
  const stride = Math.max(1, Math.floor(geometry.x.length / 36));

  for (let rowIndex = 0; rowIndex < geometry.y.length; rowIndex += stride) {
    for (let columnIndex = 0; columnIndex < geometry.x.length; columnIndex += stride) {
      const objective = geometry.objective[rowIndex]?.[columnIndex];
      if (geometry.feasible[rowIndex]?.[columnIndex] && objective !== null && Number.isFinite(objective)) {
        x.push(geometry.x[columnIndex]);
        y.push(geometry.y[rowIndex]);
        z.push(objective);
      }
    }
  }

  return { x, y, z };
}

function objectiveAtNearest(geometry: GeometryResponse, xValue: number, yValue: number) {
  const xIndex = nearestIndex(geometry.x, xValue);
  const yIndex = nearestIndex(geometry.y, yValue);
  return geometry.objective[yIndex]?.[xIndex] ?? 0;
}

function nearestIndex(values: number[], target: number) {
  let bestIndex = 0;
  let bestDistance = Number.POSITIVE_INFINITY;

  values.forEach((value, index) => {
    const distance = Math.abs(value - target);
    if (distance < bestDistance) {
      bestIndex = index;
      bestDistance = distance;
    }
  });

  return bestIndex;
}

function viewDescription(plotView: PlotView) {
  if (plotView === "actual-graph") {
    return "Objective values as a colored field with the feasible region and constraints overlaid.";
  }
  if (plotView === "surface-3d") {
    return "The sampled objective surface, with feasible points and solver trace lifted into 3D.";
  }
  return "Default view: high-contrast objective level curves over the feasible region.";
}

function HistoryPlot({ Plot, result }: { Plot: PlotComponent; result: SolveResponse }) {
  const objectiveHistory = result.history.objective;
  const violationHistory = result.history.violation;
  const eigenvalues = result.history.eigenvalues;
  const traces: Data[] = [];

  if (objectiveHistory?.length) {
    traces.push({
      type: "scatter",
      mode: "lines+markers",
      x: objectiveHistory.map((_, index) => index),
      y: objectiveHistory,
      name: "Objective",
      line: { color: "#4c6fff", width: 3 },
    });
  }

  if (violationHistory?.length) {
    traces.push({
      type: "scatter",
      mode: "lines+markers",
      x: violationHistory.map((_, index) => index),
      y: violationHistory,
      yaxis: "y2",
      name: "Violation",
      line: { color: "#e05263", width: 2 },
    });
  }

  if (eigenvalues?.length) {
    traces.push({
      type: "bar",
      x: eigenvalues.map((_, index) => `lambda ${index + 1}`),
      y: eigenvalues,
      name: "PSD eigenvalues",
      marker: { color: "#43aa8b" },
    });
  }

  if (traces.length === 0) {
    return null;
  }

  return (
    <Plot
      className="history-plot"
      data={traces}
      layout={{
        height: 260,
        margin: { l: 48, r: 48, t: 20, b: 44 },
        paper_bgcolor: "#ffffff",
        plot_bgcolor: "#ffffff",
        font: { color: PLOT_TEXT_COLOR },
        xaxis: planarAxis("iteration"),
        yaxis: planarAxis("Objective / eigenvalue"),
        yaxis2: { ...planarAxis("Violation"), overlaying: "y", side: "right", showgrid: false },
        legend: { orientation: "h", y: -0.22, font: { color: PLOT_TEXT_COLOR } },
      }}
      config={{ responsive: true, displayModeBar: false }}
    />
  );
}

function usePlotlyComponent() {
  const [Plot, setPlot] = useState<PlotComponent | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    Promise.all([import("plotly.js-dist-min"), import("react-plotly.js/factory")])
      .then(([plotlyModule, factoryModule]) => {
        if (cancelled) {
          return;
        }
        const factoryDefault = factoryModule.default as unknown as
          | ((plotly: unknown) => PlotComponent)
          | { default: (plotly: unknown) => PlotComponent };
        const createPlotlyComponent =
          typeof factoryDefault === "function" ? factoryDefault : factoryDefault.default;
        const plotly =
          (plotlyModule as { default?: unknown }).default ?? (plotlyModule as unknown);
        setPlot(() => createPlotlyComponent(plotly) as PlotComponent);
      })
      .catch((caught: Error) => {
        if (!cancelled) {
          setError(caught.message);
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  return { Plot, error };
}
