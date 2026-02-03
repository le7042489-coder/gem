"use client";

import type { PointerEvent, WheelEvent } from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { PredictResponse, Box, Finding } from "@/lib/api";
import { jsPDF } from "jspdf";

const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

export default function WorkbenchPage() {
  const [data, setData] = useState<PredictResponse | null>(null);
  const [report, setReport] = useState("");
  const [scale, setScale] = useState(1);
  const [translate, setTranslate] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const dragStart = useRef({ x: 0, y: 0 });
  const translateStart = useRef({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const stored = sessionStorage.getItem("gem_analysis");
    if (stored) {
      const parsed: PredictResponse = JSON.parse(stored);
      setData(parsed);
      setReport(parsed.report || "");
    }
  }, []);

  const imageSrc = useMemo(() => {
    if (!data?.image_base64) return null;
    return `data:image/jpeg;base64,${data.image_base64}`;
  }, [data]);

  const boxes = data?.boxes || [];
  const findings = data?.findings || [];

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

  const focusOnFinding = (finding: Finding, box?: Box) => {
    if (!containerRef.current || !data) return;
    const rect = containerRef.current.getBoundingClientRect();
    const imageWidth = data.image_width || rect.width;
    const imageHeight = data.image_height || rect.height;
    const focusScale = 1.8;
    let centerX = imageWidth / 2;
    let centerY = imageHeight / 2;

    if (box) {
      centerX = (box.x1 + box.x2) / 2 * imageWidth;
      centerY = (box.y1 + box.y2) / 2 * imageHeight;
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

  if (!data) {
    return (
      <div className="card">
        <h2>No analysis data</h2>
        <p className="muted">Please run an analysis from the landing page first.</p>
        <Link className="button-primary" href="/">Go to Upload</Link>
      </div>
    );
  }

  return (
    <div>
      <div className="topbar">
        <div>
          <h1>Diagnosis Workbench</h1>
          <p className="muted">Interactive ECG evidence + clinical report editor.</p>
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
              aspectRatio: data.image_width && data.image_height ? `${data.image_width} / ${data.image_height}` : "16 / 9"
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
                height: "100%"
              }}
            >
              {imageSrc && (
                <img src={imageSrc} alt="ECG" className="viewer-image" />
              )}

              {boxes.map((box) => (
                <div
                  key={box.index}
                  className="overlay-box"
                  style={{
                    left: `${box.x1 * 100}%`,
                    top: `${box.y1 * 100}%`,
                    width: `${(box.x2 - box.x1) * 100}%`,
                    height: `${(box.y2 - box.y1) * 100}%`
                  }}
                >
                  <span className="overlay-label">{box.label}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="card" style={{ display: "flex", flexDirection: "column", gap: 18 }}>
          <div>
            <h2>AI Findings</h2>
            <p className="muted">Click to focus on the evidence region.</p>
            <div style={{ display: "flex", flexDirection: "column", gap: 12, marginTop: 12 }}>
              {findings.length === 0 && (
                <p className="muted">No structured findings were returned.</p>
              )}
              {findings.map((finding) => {
                const box = boxes.find((item) => item.index === finding.index);
                return (
                  <div
                    key={finding.index}
                    className="finding-item"
                    onClick={() => focusOnFinding(finding, box)}
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
