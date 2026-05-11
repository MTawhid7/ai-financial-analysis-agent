import { Suspense, lazy, type ComponentType } from "react";
import type { Layout, Data, Config } from "plotly.js";

interface PlotProps {
  data: Data[];
  layout?: Partial<Layout>;
  config?: Partial<Config>;
  style?: React.CSSProperties;
  useResizeHandler?: boolean;
}

// react-plotly.js is a CommonJS module. Vite may resolve it as { default: Component }
// or as the module itself. The `?? mod` fallback handles builds where mod.default
// is undefined — without it React.lazy throws "Element type is invalid".
const Plot = lazy<ComponentType<PlotProps>>(() =>
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  import("react-plotly.js").then((mod: any) => ({
    default: mod.default?.default || mod.default || mod
  }))
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
            ...(figure.layout as Partial<Layout>),
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
