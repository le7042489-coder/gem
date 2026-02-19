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

export type SampleItem = {
  id: string;
  source: string;
  image: string;
  time_series: string;
  mask_path: string | null;
};

export type SampleListResponse = {
  total: number;
  offset: number;
  limit: number;
  items: SampleItem[];
};

export type EvidenceItem = {
  id: string;
  lead: string;
  t_start_ms: number;
  t_end_ms: number;
  measurement_name: string;
  value: number;
  unit: string;
  source: string;
  quality: string;
};

export type StructuredSection = {
  text: string;
  evidence_ids: string[];
};

export type StructuredReport = {
  rhythm: StructuredSection;
  conduction: StructuredSection;
  st_t: StructuredSection;
  axis: StructuredSection;
  summary: string;
};

export type PredictPlusResponse = {
  sample_id: string;
  structured: StructuredReport;
  evidence: EvidenceItem[];
  raw_model_output: string;
  parser_status: string;
  context_used: Record<string, unknown>;
  fs_used: number;
  preprocess: {
    fs_original: number;
    len_original: number;
    resampled: boolean;
    resample_method: string;
    cropped: boolean;
    padded: boolean;
  };
  viewer: {
    image_id: string;
    layout: string;
    total_ms: number;
    segment_ms: number;
  };
  validation_warnings: string[];
};

export type PatientContext = {
  age?: string | number | null;
  sex?: string | null;
  encounter?: string | null;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || "API request failed");
  }
  return response.json();
}

export function getImageUrl(imageId: string): string {
  return `${API_BASE}/images/${encodeURIComponent(imageId)}`;
}

export async function listSamples(params?: {
  source?: string;
  q?: string;
  offset?: number;
  limit?: number;
}): Promise<SampleListResponse> {
  const search = new URLSearchParams();
  if (params?.source) search.set("source", params.source);
  if (params?.q) search.set("q", params.q);
  if (typeof params?.offset === "number") search.set("offset", String(params.offset));
  if (typeof params?.limit === "number") search.set("limit", String(params.limit));
  const query = search.toString();
  const response = await fetch(`${API_BASE}/samples${query ? `?${query}` : ""}`, {
    method: "GET",
  });
  return parseResponse<SampleListResponse>(response);
}

export async function predictPlusById(input: {
  sample_id: string;
  query?: string;
  patient_context?: PatientContext;
}): Promise<PredictPlusResponse> {
  const response = await fetch(`${API_BASE}/predict_plus/by_id`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  return parseResponse<PredictPlusResponse>(response);
}

export async function predictPlusUpload(input: {
  time_series_file: File;
  image_file?: File | null;
  query?: string;
  patient_context?: PatientContext;
  fs_original?: number;
}): Promise<PredictPlusResponse> {
  const formData = new FormData();
  formData.append("time_series_file", input.time_series_file);
  if (input.image_file) {
    formData.append("image_file", input.image_file);
  }
  if (input.query) {
    formData.append("query", input.query);
  }
  if (input.patient_context) {
    formData.append("patient_context_json", JSON.stringify(input.patient_context));
  }
  if (typeof input.fs_original === "number") {
    formData.append("fs_original", String(input.fs_original));
  }

  const response = await fetch(`${API_BASE}/predict_plus/upload`, {
    method: "POST",
    body: formData,
  });
  return parseResponse<PredictPlusResponse>(response);
}

export async function predict(files: File[]): Promise<PredictResponse> {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));

  const response = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    body: formData,
  });
  return parseResponse<PredictResponse>(response);
}

