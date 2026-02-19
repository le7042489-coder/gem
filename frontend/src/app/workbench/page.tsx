"use client";

import type { PointerEvent, WheelEvent } from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { jsPDF } from "jspdf";
import {
  Box,
  Finding,
  getImageUrl,
  PredictPlusResponse,
  PredictResponse,
} from "@/lib/api";

const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

const LEAD_LAYOUT: Record<string, { row: number; col: number; offsetMs: number }> = {
  I: { row: 0, col: 0, offsetMs: 0 },
  aVR: { row: 0, col: 1, offsetMs: 2500 },
  V1: { row: 0, col: 2, offsetMs: 5000 },
  V4: { row: 0, col: 3, offsetMs: 7500 },
  II: { row: 1, col: 0, offsetMs: 0 },
  aVL: { row: 1, col: 1, offsetMs: 2500 },
  V2: { row: 1, col: 2, offsetMs: 5000 },
  V5: { row: 1, col: 3, offsetMs: 7500 },
  III: { row: 2, col: 0, offsetMs: 0 },
  aVF: { row: 2, col: 1, offsetMs: 2500 },
  V3: { row: 2, col: 2, offsetMs: 5000 },
  V6: { row: 2, col: 3, offsetMs: 7500 },
};

type Overlay = {
  id: string;
  label: string;
  x1: number;
  x2: number;
  y1: number;
  y2: number;
};

function isPlusData(data: unknown): data is PredictPlusResponse {
  if (!data || typeof data !== "object") return false;
  const candidate = data as Partial<PredictPlusResponse>;
  return !!candidate.viewer && !!candidate.structured && Array.isArray(candidate.evidence);
}

function buildOverlayFromEvidence(data: PredictPlusResponse): Overlay[] {
  const totalMs = data.viewer.total_ms || 10000;
  const segmentMs = data.viewer.segment_ms || 2500;
  const overlays: Overlay[] = [];
  for (const ev of data.evidence) {
    const pos = LEAD_LAYOUT[ev.lead];
    if (!pos) continue;
    const start = clamp(ev.t_start_ms, 0, totalMs);
    const end = clamp(ev.t_end_ms, 0, totalMs);
    const relStart = clamp(start - pos.offsetMs, 0, segmentMs);
    const relEnd = clamp(end - pos.offsetMs, 0, segmentMs);
    if (relEnd <= relStart) continue;
    overlays.push({
      id: ev.id,
      label: ev.id,
      x1: (pos.col + relStart / segmentMs) / 4,
      x2: (pos.col + relEnd / segmentMs) / 4,
      y1: pos.row / 3,
      y2: (pos.row + 1) / 3,
    });
  }
  return overlays;
}

function buildDefaultReport(data: PredictPlusResponse): string {
  const s = data.structured;
  return [
    `Summary: ${s.summary || ""}`,
    "",
    `Rhythm: ${s.rhythm.text || ""}`,
    `Conduction: ${s.conduction.text || ""}`,
    `ST-T: ${s.st_t.text || ""}`,
    `Axis: ${s.axis.text || ""}`,
  ].join("\n");
}

export default function WorkbenchPage() {
  const [plusData, setPlusData] = useState<PredictPlusResponse | null>(null);
  const [legacyData, setLegacyData] = useState<PredictResponse | null>(null);
  const [report, setReport] = useState("");
  const [scale, setScale] = useState(1);
  const [translate, setTranslate] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const dragStart = useRef({ x: 0, y: 0 });
  const translateStart = useRef({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const plusRaw = sessionStorage.getItem("gem_analysis_v2");
    if (plusRaw) {
      try {
        const parsed = JSON.parse(plusRaw);
        if (isPlusData(parsed)) {
          setPlusData(parsed);
          setLegacyData(null);
          setReport(buildDefaultReport(parsed));
          return;
        }
      } catch {
        // ignore parse issue and fallback to legacy.
      }
    }

    const legacyRaw = sessionStorage.getItem("gem_analysis");
    if (legacyRaw) {
      const parsed: PredictResponse = JSON.parse(legacyRaw);
      setLegacyData(parsed);
      setPlusData(null);
      setReport(parsed.report || "");
    }
  }, []);

  const imageSrc = useMemo(() => {
    if (plusData) return getImageUrl(plusData.viewer.image_id);
    if (legacyData?.image_base64) return `data:image/jpeg;base64,${legacyData.image_base64}`;
    return null;
  }, [plusData, legacyData]);

  const overlays = useMemo(() => {
    if (plusData) return buildOverlayFromEvidence(plusData);
    if (!legacyData) return [];
    return legacyData.boxes.map((box) => ({
      id: String(box.index),
      label: box.label,
      x1: box.x1,
      x2: box.x2,
      y1: box.y1,
      y2: box.y2,
    }));
  }, [plusData, legacyData]);

  const handleWheel = (event: WheelEvent<HTMLDivElement>) => {
    event.preventDefault();
    const delta = event.deltaY < 0 ? 0.1 : -0.1;
    setScale((prev) => clamp(prev + delta, 1, 4));
  };

  const handlePointerDown = (event: PointerEvent<HTMLDivElement>) => {
    setDragging(true);
    dragStart.current = { x: event.clientX, y: event.clientY };
    translateStart.current = { ...translate };
  };

  const handlePointerMove = (event: PointerEvent<HTMLDivElement>) => {
    if (!dragging) return;
    const dx = event.clientX - dragStart.current.x;
    const dy = event.clientY - dragStart.current.y;
    setTranslate({ x: translateStart.current.x + dx, y: translateStart.current.y + dy });
  };

  const handlePointerUp = () => {
    setDragging(false);
  };

  const focusOnOverlay = (overlay?: Overlay) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const focusScale = 1.8;
    let centerX = rect.width / 2;
    let centerY = rect.height / 2;
    if (overlay) {
      centerX = ((overlay.x1 + overlay.x2) / 2) * rect.width;
      centerY = ((overlay.y1 + overlay.y2) / 2) * rect.height;
    }
    const targetX = rect.width / 2 - centerX * focusScale;
    const targetY = rect.height / 2 - centerY * focusScale;
    setScale(focusScale);
    setTranslate({ x: targetX, y: targetY });
  };

  const resetView = () => {
    setScale(1);
    setTranslate({ x: 0, y: 0 });
  };

  const exportPdf = () => {
    const doc = new jsPDF();
    const lines = doc.splitTextToSize(report, 180);
    doc.text(lines, 10, 20);
    doc.save("ecg-report.pdf");
  };

  if (!plusData && !legacyData) {
    return (
      <div className="card">
        <h2>No analysis data</h2>
        <p className="muted">Please run an analysis from the landing page first.</p>
        <Link className="button-primary" href="/">Go to Upload</Link>
      </div>
    );
  }

  const width = legacyData?.image_width || 1600;
  const height = legacyData?.image_height || 1200;

  return (
    <div>
      <div className="topbar">
        <div>
          <h1>Diagnosis Workbench</h1>
          <p className="muted">Grounded ECG interpretation with evidence-linked review.</p>
        </div>
        <Link className="button-secondary" href="/">New Analysis</Link>
      </div>

      <div className="workbench">
        <div className="card">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <h2>Evidence Viewer</h2>
            <button className="button-secondary" onClick={resetView}>Reset View</button>
          </div>

          <div
            ref={containerRef}
            className="viewer-shell"
            style={{
              marginTop: 16,
              width: "100%",
              aspectRatio: `${width} / ${height}`,
            }}
            onWheel={handleWheel}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerLeave={handlePointerUp}
          >
            <div
              className="viewer-inner"
              style={{
                transform: `translate(${translate.x}px, ${translate.y}px) scale(${scale})`,
                width: "100%",
                height: "100%",
              }}
            >
              {imageSrc && <img src={imageSrc} alt="ECG" className="viewer-image" />}

              {overlays.map((box) => (
                <div
                  key={box.id}
                  className="overlay-box"
                  style={{
                    left: `${box.x1 * 100}%`,
                    top: `${box.y1 * 100}%`,
                    width: `${(box.x2 - box.x1) * 100}%`,
                    height: `${(box.y2 - box.y1) * 100}%`,
                  }}
                >
                  <span className="overlay-label">{box.label}</span>
                </div>
              ))}
            </div>
          </div>

          {plusData && (
            <div style={{ marginTop: 14 }}>
              <p className="muted">
                fs_used={plusData.fs_used}, fs_original={plusData.preprocess.fs_original}, len_original={plusData.preprocess.len_original},
                resampled={String(plusData.preprocess.resampled)}
              </p>
            </div>
          )}
        </div>

        <div className="card" style={{ display: "flex", flexDirection: "column", gap: 18 }}>
          {plusData ? (
            <>
              <div>
                <h2>Structured Report</h2>
                {(["rhythm", "conduction", "st_t", "axis"] as const).map((key) => {
                  const section = plusData.structured[key];
                  return (
                    <div key={key} className="finding-item" style={{ marginTop: 10 }}>
                      <strong>{key.toUpperCase()}</strong>
                      <div className="muted">{section.text || "(empty)"}</div>
                      <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                        {section.evidence_ids.length === 0 && <span className="muted">No evidence ids</span>}
                        {section.evidence_ids.map((evId) => (
                          <button
                            key={evId}
                            className="button-secondary"
                            style={{ padding: "4px 10px" }}
                            onClick={() => focusOnOverlay(overlays.find((x) => x.id === evId))}
                          >
                            {evId}
                          </button>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>

              <div>
                <h2>Evidence Table</h2>
                <div style={{ display: "flex", flexDirection: "column", gap: 8, marginTop: 8 }}>
                  {plusData.evidence.map((ev) => (
                    <div
                      key={ev.id}
                      className="finding-item"
                      onClick={() => focusOnOverlay(overlays.find((x) => x.id === ev.id))}
                    >
                      <div style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
                        <strong>{ev.id}</strong>
                        <span className="muted">{ev.source}/{ev.quality}</span>
                      </div>
                      <div className="muted">
                        {ev.measurement_name}: {ev.value} {ev.unit}
                      </div>
                      <div className="muted">
                        Lead {ev.lead} | {ev.t_start_ms.toFixed(1)}-{ev.t_end_ms.toFixed(1)} ms
                      </div>
                    </div>
                  ))}
                </div>
                {plusData.validation_warnings.length > 0 && (
                  <div style={{ marginTop: 10 }}>
                    <p className="muted">Validation warnings:</p>
                    <ul>
                      {plusData.validation_warnings.map((w) => <li key={w}>{w}</li>)}
                    </ul>
                  </div>
                )}
              </div>
            </>
          ) : (
            <div>
              <h2>AI Findings (Legacy)</h2>
              <p className="muted">Click to focus on the evidence region.</p>
              <div style={{ display: "flex", flexDirection: "column", gap: 12, marginTop: 12 }}>
                {(legacyData?.findings || []).length === 0 && (
                  <p className="muted">No structured findings were returned.</p>
                )}
                {(legacyData?.findings || []).map((finding: Finding) => {
                  const box = (legacyData?.boxes || []).find((item: Box) => item.index === finding.index);
                  const overlay = box
                    ? {
                        id: String(box.index),
                        label: box.label,
                        x1: box.x1,
                        x2: box.x2,
                        y1: box.y1,
                        y2: box.y2,
                      }
                    : undefined;
                  return (
                    <div
                      key={finding.index}
                      className="finding-item"
                      onClick={() => focusOnOverlay(overlay)}
                    >
                      <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                        <span className="finding-badge">{finding.label}</span>
                        <strong>{finding.symptom}</strong>
                      </div>
                      <div className="muted">Lead: {finding.lead}</div>
                      <div className="muted">Time: {finding.time_range}</div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          <div>
            <h2>Editable Report</h2>
            <textarea
              className="report-textarea"
              value={report}
              onChange={(event) => setReport(event.target.value)}
            />
            <div style={{ marginTop: 12 }}>
              <button className="button-primary" onClick={exportPdf}>Export PDF</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

