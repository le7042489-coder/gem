"use client";

import { useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  listSamples,
  predict,
  predictPlusById,
  predictPlusUpload,
  PredictResponse,
  PredictPlusResponse,
  SampleItem,
} from "@/lib/api";

type Mode = "sample" | "upload" | "legacy";

export default function LandingPage() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [mode, setMode] = useState<Mode>("sample");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const [samples, setSamples] = useState<SampleItem[]>([]);
  const [search, setSearch] = useState("");
  const [selectedSampleId, setSelectedSampleId] = useState<string | null>(null);

  const [npyFile, setNpyFile] = useState<File | null>(null);
  const [pngFile, setPngFile] = useState<File | null>(null);
  const [fsOriginal, setFsOriginal] = useState("");

  const [legacyFiles, setLegacyFiles] = useState<File[]>([]);

  const selectedSample = useMemo(
    () => samples.find((item) => item.id === selectedSampleId) || null,
    [samples, selectedSampleId]
  );

  const loadSampleList = async (query?: string) => {
    const resp = await listSamples({ q: query || "", limit: 80, offset: 0 });
    setSamples(resp.items);
    if (resp.items.length > 0) {
      setSelectedSampleId((prev) => prev || resp.items[0].id);
    }
  };

  useEffect(() => {
    loadSampleList().catch((err) => {
      setError(err instanceof Error ? err.message : "Failed to load samples");
    });
  }, []);

  const toWorkbenchPlus = (data: PredictPlusResponse) => {
    sessionStorage.setItem("gem_analysis_v2", JSON.stringify(data));
    sessionStorage.removeItem("gem_analysis");
    router.push("/workbench");
  };

  const toWorkbenchLegacy = (data: PredictResponse) => {
    sessionStorage.setItem("gem_analysis", JSON.stringify(data));
    sessionStorage.removeItem("gem_analysis_v2");
    router.push("/workbench");
  };

  const runSampleMode = async () => {
    if (!selectedSampleId) {
      setError("Please select a sample.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const data = await predictPlusById({ sample_id: selectedSampleId });
      toWorkbenchPlus(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  };

  const runUploadMode = async () => {
    if (!npyFile) {
      setError("Please upload a .npy time series file.");
      return;
    }
    const fs = fsOriginal.trim() ? Number(fsOriginal.trim()) : undefined;
    if (fsOriginal.trim() && Number.isNaN(fs)) {
      setError("fs_original must be a valid number.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const data = await predictPlusUpload({
        time_series_file: npyFile,
        image_file: pngFile,
        fs_original: fs,
      });
      toWorkbenchPlus(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  };

  const handleLegacyFiles = (fileList: FileList | null) => {
    if (!fileList) return;
    const list = Array.from(fileList).filter((file) => {
      const name = file.name.toLowerCase();
      return name.endsWith(".dat") || name.endsWith(".hea");
    });
    setLegacyFiles(list);
    setError(null);
  };

  const runLegacyMode = async () => {
    if (legacyFiles.length === 0) {
      setError("Please upload .dat and .hea files.");
      return;
    }
    const hasDat = legacyFiles.some((file) => file.name.toLowerCase().endsWith(".dat"));
    const hasHea = legacyFiles.some((file) => file.name.toLowerCase().endsWith(".hea"));
    if (!hasDat || !hasHea) {
      setError("Both .dat and .hea files are required.");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const data = await predict(legacyFiles);
      toWorkbenchLegacy(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid-2col">
      <div className="card">
        <h1>GEM+ ECG Workbench</h1>
        <p className="muted">Grounded ECG report with lead/time/measurement evidence.</p>

        <div style={{ display: "flex", gap: 8, marginTop: 14, marginBottom: 14 }}>
          <button
            className={mode === "sample" ? "button-primary" : "button-secondary"}
            onClick={() => setMode("sample")}
            disabled={loading}
          >
            Local Samples
          </button>
          <button
            className={mode === "upload" ? "button-primary" : "button-secondary"}
            onClick={() => setMode("upload")}
            disabled={loading}
          >
            Upload .npy/.png
          </button>
          <button
            className={mode === "legacy" ? "button-primary" : "button-secondary"}
            onClick={() => setMode("legacy")}
            disabled={loading}
          >
            Legacy .dat/.hea
          </button>
        </div>

        {mode === "sample" && (
          <div>
            <div style={{ display: "flex", gap: 8, marginBottom: 10 }}>
              <input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search sample id/source..."
                className="report-textarea"
                style={{ minHeight: 42, resize: "none" }}
              />
              <button
                className="button-secondary"
                onClick={() => loadSampleList(search)}
                disabled={loading}
              >
                Search
              </button>
            </div>
            <div style={{ maxHeight: 260, overflowY: "auto", border: "1px solid var(--border)", borderRadius: 10 }}>
              {samples.map((item) => (
                <label
                  key={item.id}
                  style={{
                    display: "flex",
                    gap: 10,
                    alignItems: "center",
                    padding: "10px 12px",
                    borderBottom: "1px solid var(--border)",
                    cursor: "pointer",
                    background: selectedSampleId === item.id ? "#eef6ff" : "transparent",
                  }}
                >
                  <input
                    type="radio"
                    name="sample"
                    checked={selectedSampleId === item.id}
                    onChange={() => setSelectedSampleId(item.id)}
                  />
                  <span>{item.id}</span>
                  <span className="muted">({item.source})</span>
                </label>
              ))}
              {samples.length === 0 && <p className="muted" style={{ padding: 12 }}>No samples found.</p>}
            </div>
            {selectedSample && (
              <p className="muted" style={{ marginTop: 10 }}>
                Selected: {selectedSample.id} | {selectedSample.time_series}
              </p>
            )}
            <div style={{ marginTop: 14 }}>
              <button className="button-primary" onClick={runSampleMode} disabled={loading}>
                {loading ? "Analyzing..." : "Analyze Selected Sample"}
              </button>
            </div>
          </div>
        )}

        {mode === "upload" && (
          <div>
            <div className="upload-zone" style={{ textAlign: "left" }}>
              <p><strong>Time Series (.npy)</strong></p>
              <input
                type="file"
                accept=".npy"
                onChange={(e) => setNpyFile(e.target.files?.[0] || null)}
              />
              <p style={{ marginTop: 12 }}><strong>Optional Original ECG Image (.png)</strong></p>
              <input
                type="file"
                accept=".png"
                onChange={(e) => setPngFile(e.target.files?.[0] || null)}
              />
              <p style={{ marginTop: 12 }}><strong>Optional fs_original</strong></p>
              <input
                value={fsOriginal}
                onChange={(e) => setFsOriginal(e.target.value)}
                placeholder="e.g. 500 or 100"
                className="report-textarea"
                style={{ minHeight: 42, resize: "none" }}
              />
            </div>
            <div style={{ marginTop: 14 }}>
              <button className="button-primary" onClick={runUploadMode} disabled={loading}>
                {loading ? "Analyzing..." : "Analyze Uploaded ECG"}
              </button>
            </div>
          </div>
        )}

        {mode === "legacy" && (
          <div>
            <div
              className="upload-zone"
              onDragOver={(event) => event.preventDefault()}
              onDrop={(event) => {
                event.preventDefault();
                handleLegacyFiles(event.dataTransfer.files);
              }}
              role="button"
              tabIndex={0}
              onClick={() => inputRef.current?.click()}
            >
              <p><strong>Drag & Drop</strong> legacy files here</p>
              <p className="muted">.dat and .hea files only</p>
              <input
                type="file"
                accept=".dat,.hea"
                multiple
                ref={inputRef}
                style={{ display: "none" }}
                onChange={(event) => handleLegacyFiles(event.target.files)}
              />
              <button className="button-secondary" type="button">Browse Files</button>
            </div>
            {legacyFiles.length > 0 && (
              <ul style={{ marginTop: 10 }}>
                {legacyFiles.map((f) => <li key={f.name}>{f.name}</li>)}
              </ul>
            )}
            <div style={{ marginTop: 14 }}>
              <button className="button-primary" onClick={runLegacyMode} disabled={loading}>
                {loading ? "Analyzing..." : "Run Legacy Analysis"}
              </button>
            </div>
          </div>
        )}

        {error && <p style={{ color: "#ef4444", marginTop: 12 }}>{error}</p>}
      </div>

      <div className="card">
        <h2>What this version fixes</h2>
        <ul className="muted" style={{ lineHeight: 1.8 }}>
          <li>Signal preprocessing reports fs/length/resample metadata</li>
          <li>Single coordinate system: model-viewer both use standard 3x4 rendering</li>
          <li>Light response payload using image_id + on-demand image fetch</li>
          <li>Evidence id validation to avoid broken references</li>
        </ul>
      </div>
    </div>
  );
}

