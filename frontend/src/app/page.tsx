"use client";

import { useRouter } from "next/navigation";
import type { DragEvent } from "react";
import { useRef, useState } from "react";
import { predict, PredictResponse } from "@/lib/api";

export default function LandingPage() {
  const router = useRouter();
  const [files, setFiles] = useState<File[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const handleFiles = (fileList: FileList | null) => {
    if (!fileList) return;
    const list = Array.from(fileList);
    setFiles(list);
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    handleFiles(event.dataTransfer.files);
  };

  const startAnalysis = async () => {
    setError(null);
    if (files.length === 0) {
      setError("Please upload .dat and .hea files.");
      return;
    }
    setLoading(true);
    try {
      const data: PredictResponse = await predict(files);
      if (data.report?.startsWith("❌") || data.report?.startsWith("Error")) {
        setError(data.report);
        return;
      }
      sessionStorage.setItem("gem_analysis", JSON.stringify(data));
      router.push("/workbench");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid-2col">
      <div className="card">
        <h1>Clinical ECG Diagnosis Workbench</h1>
        <p className="muted">
          Upload raw <strong>.dat</strong> and <strong>.hea</strong> files to start AI-assisted analysis.
        </p>

        <div
          className="upload-zone"
          onDragOver={(event) => event.preventDefault()}
          onDrop={handleDrop}
          role="button"
          tabIndex={0}
          onClick={() => inputRef.current?.click()}
        >
          <p><strong>Drag & Drop</strong> your ECG files here</p>
          <p className="muted">.dat and .hea files only</p>
          <input
            type="file"
            accept=".dat,.hea"
            multiple
            ref={inputRef}
            style={{ display: "none" }}
            onChange={(event) => handleFiles(event.target.files)}
          />
          <button className="button-secondary" type="button">Browse Files</button>
        </div>

        <div style={{ marginTop: 18, display: "flex", gap: 12, alignItems: "center" }}>
          <button className="button-primary" onClick={startAnalysis} disabled={loading}>
            {loading ? "Analyzing..." : "Start AI Analysis"}
          </button>
          <button className="button-secondary" disabled>
            Load Demo Data
          </button>
        </div>

        {files.length > 0 && (
          <div style={{ marginTop: 16 }}>
            <p className="muted">Selected files:</p>
            <ul>
              {files.map((file) => (
                <li key={file.name}>{file.name}</li>
              ))}
            </ul>
          </div>
        )}

        {error && <p style={{ color: "#ef4444", marginTop: 12 }}>{error}</p>}
      </div>

      <div className="card">
        <h2>What you will get</h2>
        <ul className="muted" style={{ lineHeight: 1.8 }}>
          <li>Structured findings with lead & time annotations</li>
          <li>Visual evidence on ECG grid with anomaly overlays</li>
          <li>Editable clinical report & export-ready summary</li>
        </ul>
      </div>
    </div>
  );
}
