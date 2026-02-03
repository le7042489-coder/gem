export type Finding = {
  index: number;
  label: string;
  symptom: string;
  lead: string;
  start: number | null;
  end: number | null;
  time_range: string;
};

export type Box = {
  index: number;
  label: string;
  x1: number;
  x2: number;
  y1: number;
  y2: number;
};

export type PredictResponse = {
  report: string;
  findings: Finding[];
  boxes: Box[];
  image_base64: string;
  image_width: number;
  image_height: number;
  raw_output: string;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export async function predict(files: File[]): Promise<PredictResponse> {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));

  const response = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || "API request failed");
  }

  return response.json();
}
