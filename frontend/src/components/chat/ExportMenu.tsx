import { useEffect, useState } from "react";
import { exportPdf, exportDocx, exportXlsx, getExportAvailable } from "../../lib/api";

interface Props {
  reportId: string;
}

interface Availability {
  pdf: boolean;
  docx: boolean;
  xlsx: boolean;
}

export function ExportMenu({ reportId }: Props) {
  const [availability, setAvailability] = useState<Availability>({ pdf: false, docx: true, xlsx: true });
  const [loading, setLoading] = useState<string | null>(null);

  useEffect(() => {
    getExportAvailable().then(setAvailability).catch(() => {});
  }, []);

  const download = async (format: "pdf" | "docx" | "xlsx") => {
    setLoading(format);
    try {
      let blob: Blob;
      let filename: string;

      if (format === "pdf") {
        blob = await exportPdf(reportId);
        filename = `report_${reportId.slice(0, 8)}.pdf`;
      } else if (format === "docx") {
        blob = await exportDocx(reportId);
        filename = `report_${reportId.slice(0, 8)}.docx`;
      } else {
        blob = await exportXlsx(reportId);
        filename = `report_${reportId.slice(0, 8)}.xlsx`;
      }

      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      alert(`Export failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setLoading(null);
    }
  };

  const buttons: Array<{ key: "pdf" | "docx" | "xlsx"; label: string; available: boolean }> = [
    { key: "pdf",  label: "PDF",  available: availability.pdf },
    { key: "docx", label: "Word", available: availability.docx },
    { key: "xlsx", label: "Excel (live formulas)", available: availability.xlsx },
  ];

  return (
    <div className="flex flex-wrap items-center gap-2 mt-3 pt-3 border-t border-zinc-700">
      <span className="text-xs text-zinc-500">Export:</span>
      {buttons.map(({ key, label, available }) => (
        <button
          key={key}
          onClick={() => download(key)}
          disabled={!available || loading !== null}
          title={available ? `Download as ${label}` : `${label} export not available on this server`}
          className={`
            text-xs px-2.5 py-1 rounded border transition-colors
            ${available
              ? "border-zinc-600 text-zinc-300 hover:border-violet-500 hover:text-violet-300"
              : "border-zinc-800 text-zinc-700 cursor-not-allowed"
            }
            ${loading === key ? "opacity-60" : ""}
          `}
        >
          {loading === key ? "…" : label}
        </button>
      ))}
    </div>
  );
}
