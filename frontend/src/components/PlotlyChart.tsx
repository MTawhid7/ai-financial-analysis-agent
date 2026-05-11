import { Suspense, lazy } from "react";
import type { Layout, Data } from "plotly.js";

// react-plotly.js is a CommonJS module. Without the .then() normalisation,
// Vite's ESM dynamic import resolves to the namespace object { default, ... }
// and React.lazy crashes with "Element type is invalid" on first load.
const Plot = lazy(() =>
  import("react-plotly.js").then((mod) => ({ default: mod.default }))
);

interface Props {
  figure: { data: object[]; layout: object };
  title?: string;
}

export function PlotlyChart({ figure, title }: Props) {
  return (
    <div className="rounded-lg overflow-hidden border border-zinc-700 my-3">
      {title && (
        <p className="text-xs font-medium text-zinc-400 px-3 pt-2">{title}</p>
      )}
      <Suspense
        fallback={
          <div className="h-64 flex items-center justify-center text-zinc-500 text-sm">
            Loading chart…
          </div>
        }
      >
        <Plot
          data={figure.data as Data[]}
          layout={{
            ...figure.layout as Partial<Layout>,
            autosize: true,
            height: 280,
            margin: { l: 50, r: 20, t: 40, b: 50 },
          }}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: "100%" }}
          useResizeHandler
        />
      </Suspense>
    </div>
  );
}
