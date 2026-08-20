
IT Incident Triaging & Root-Cause Engine, designed to be built entirely on
your Ryzen 3 CPU and 15 GB RAM environment.﻿
1. Architecture Overview﻿
                          ┌───────────────────────────┐﻿
                          │
Incoming Incident Payload │﻿
                          └─────────────┬─────────────┘﻿
                                        │﻿
                 ┌──────────────────────┴──────────────────────┐﻿
                 ▼                                             ▼﻿
       [Unstructured Log
String]                     [Raw
Time-Series Metrics]﻿
                 │                                             │﻿
                 ▼                                             ▼﻿
  ┌──────────────────────────────┐              ┌──────────────────────────────┐﻿
  │  SmolLM2-135M (LoRA
CPU)     │              │
Advanced Feature Engineering │﻿
  │  - JSON Structured
Output    │              │
- Rolling Stats / EWMA       │﻿
  │  - Category &
Root-Cause     │              │
- FFT Spectral Peak & Energy │﻿
  │  - Confidence /
Embeddings   │              │
- Cross-Metric Interaction   │﻿
  └──────────────┬───────────────┘              └──────────────┬───────────────┘﻿
                 │                                             │﻿
                 │                                             ▼﻿
                 │                              ┌──────────────────────────────┐﻿
                 │                              │
LightGBM Anomaly Predictor   │﻿
                 │                              │
- Sub-5ms CPU Latency        │﻿
                 │                              │
- Calibrated Risk Score      │﻿
                 │                              └──────────────┬───────────────┘﻿
                 │                                             │﻿
                 └──────────────────────┬──────────────────────┘﻿
                                        ▼﻿
                         ┌─────────────────────────────┐﻿
                         │   Heuristic
Fusion Layer    │﻿
                         │
(Weighted Risk + Routing)   │﻿
                         └──────────────┬──────────────┘﻿
                                        ▼﻿
                         ┌─────────────────────────────┐﻿
                         │   FastAPI
/ Docker Engine   │﻿
                         └─────────────────────────────┘﻿
 
2. Dataset Strategy﻿
 * Log Data: Use the
open-source LogHub dataset (specifically HDFS, BGL, or Thunderbird subsets).
Extract around 5,000 to 10,000 log lines and create structured target pairs
(Log \rightarrow JSON with error_type, component, severity).﻿
 * Telemetry Data: Use
the Server Machine Dataset (SMD) or NAB (Numenta Anomaly Benchmark) for real
multi-variate machine metrics (CPU usage, memory usage, disk read/write
throughput, network packet rates).﻿
3. Step-by-Step Implementation Blueprint﻿
Phase 1: Feature Engineering Pipeline (Pure Tabular ML)﻿
 * Window
Aggregations: Calculate rolling mean, variance, min, and max across 3 window
sizes (W \in \{5\text{m}, 15\text{m}, 60\text{m}\}).﻿
 * Exponential
Smoothing: Compute Exponentially Weighted Moving Averages (EWMA) with spans of
5 and 20 steps to capture sudden gradient shifts vs. baseline drifts.﻿
 * Signal Processing
(FFT): Apply Fast Fourier Transform (scipy.fft) over a sliding 30-step window
to extract the dominant frequency and spectral energy, distinguishing healthy
periodic load from anomalous bursts.﻿
 * Interaction Ratios:
Generate derivative features such as \frac{\text{CPU I/O Wait}}{\text{CPU
User}} and \frac{\text{Bytes Transmitted}}{\text{Packets Transmitted}}.﻿
 * Model Training:
Train a LightGBM classifier with Bayesian hyperparameter optimization (optuna)
on your 8 CPU threads. Calibrate probabilities using CalibratedClassifierCV.﻿
Phase 2: Fine-Tuning the Small Language Model (SLM)﻿
 * Base Model: Choose
HuggingFaceTB/SmolLM2-135M-Instruct or Qwen/Qwen2.5-0.5B-Instruct.﻿
 * PEFT / LoRA Setup:﻿
   * Target modules: q_proj, v_proj﻿
   * Rank: r=8, Alpha: \alpha=16﻿
   * Trainable parameters: <1\% of total
weights (~1.2M parameters)﻿
 * CPU Training
Config:﻿
   * Framework: PyTorch + Hugging Face
Trainer﻿
   * Batch size: 4 (with gradient
accumulation steps = 4)﻿
   * Precision: FP32 or BF16 (if supported by
CPU)﻿
   * Epochs: 3 (approx. 20–35 minutes on 8
threads)﻿
 * Target Schema:
Force the model to output strict JSON:﻿
   {﻿
  "error_domain":
"DatabaseConnection",﻿
  "severity": "CRITICAL",﻿
  "probable_cause": "Connection
pool starvation during peak I/O"﻿
}﻿
 
Phase 3: Multimodal Decision & Fusion Layer﻿
Combine the discrete numerical risk and semantic log
extraction:﻿
 * Calculate an
aggregate Incident Triage Score (ITS):﻿
﻿
 * If \text{ITS} \ge
0.8, flag as a Priority 1 incident, generate automated mitigation instructions
from the SLM output, and route to the specific engineering on-call group.﻿
Phase 4: Optimization, API & Packaging﻿
 * Model Export:
Export the LightGBM model to native binary format and export the fine-tuned SLM
to GGUF via llama.cpp or ONNX Runtime CPU with INT8 quantization for sub-100ms
inference.﻿
 * Backend: Write a
FastAPI service with an asynchronous endpoint (/api/v1/triage).﻿
 * Containerization:
Package the complete stack into a minimal Dockerfile with multi-stage builds
and non-root execution.﻿
4. Repository Structure for Your Portfolio﻿
incident-triaging-engine/﻿
├── data/﻿
│   ├──
raw/                 #
Download scripts for LogHub & SMD﻿
│   └──
processed/           #
Processed datasets﻿
├── src/﻿
│   ├──
features/            #
FFT, EWMA, and tabular feature extraction﻿
│   ├── models/﻿
│   │   ├── train_tabular.py #
LightGBM + Optuna pipeline﻿
│   │   ├──
train_slm.py     # Hugging Face PEFT/LoRA script﻿
│   │   └──
fusion.py        # Joint scoring and
decision heuristics﻿
│   └── app/﻿
│       ├──
main.py          # FastAPI
application﻿
│       └──
schemas.py       # Pydantic input/output
validation﻿
├── benchmarks/﻿
│   └──
cpu_profile.py       # Latency (ms) and RAM
usage scripts﻿
├── Dockerfile﻿
├── requirements.txt﻿
└──
README.md                #
Architecture diagram, benchmark tables, setup steps﻿
 
 
 
Here is the exact split between your Jupyter Notebook
(Experimental Sandbox) and your IDE (Production Software System), along with
the migration bridge.﻿
Division of Labor: Notebook vs. IDE﻿
┌──────────────────────────────────────────────────────────┐﻿
│             JUPYTER
NOTEBOOK (The
Lab)                   │﻿
│  - Exploratory Data Analysis & visual signal
checks      │﻿
│  - Prototyping math transforms (FFT, EWMA
formulas)      │﻿
│  - Small-batch training trial run (sanity check
LoRA)    │﻿
│  - Exporting serialized model artifacts (.pkl,
.onnx)    │﻿
└────────────────────────────┬─────────────────────────────┘﻿
                             │﻿
                      ARTIFACTS
BRIDGE﻿
             (Saved
Model Weights & Transforms)﻿
                             │﻿
                             ▼﻿
┌──────────────────────────────────────────────────────────┐﻿
│                 IDE
/ VS CODE (The
Factory)              │﻿
│  - Stateless Feature Extraction classes (clean
.py)      │﻿
│  - Automated CLI training scripts
(train_tabular.py)     │﻿
│  - Fast inference runtime engine (ONNX /
GGUF)           │﻿
│  - FastAPI server, Pydantic input schemas &
Dockerfile   │﻿
└──────────────────────────────────────────────────────────┘﻿
 
What Happens in the Jupyter Notebook﻿
Keep 2 lightweight prototyping notebooks in a notebooks/
directory.﻿
Notebook 1: 01_signal_and_tabular_proto.ipynb﻿
 * Inspect raw data:
Load small slices of SMD/NAB metrics using pandas.﻿
 * Visual signal
sanity check: Plot raw CPU spikes vs. rolling EWMA and FFT power spectrum using
matplotlib to ensure your signal processing highlights anomalies.﻿
 * Fit baseline
LightGBM: Run a quick 5-fold cross-validation, print confusion matrices, and
verify classification metrics.﻿
Notebook 2: 02_slm_lora_sanity_check.ipynb﻿
 * Data formatting:
Load 100 sample logs from LogHub, format them into system/user/assistant prompt
templates with the target JSON output.﻿
 * Test training loop:
Run 1 epoch of LoRA fine-tuning on CPU using PEFT to verify loss convergence,
memory stability (<8 GB RAM), and gradient accumulation syntax.﻿
 * Test inference
parser: Run a quick test query and write a regex/Pydantic validation check to
ensure the model outputs valid JSON.﻿
The Bridge: Moving from Notebook Logic to Production Code﻿
Once the notebook experiments succeed, transition the core
logic into modular files in your IDE:﻿
1. Isolate Transforms into a Reusable Class﻿
 * In the notebook,
you likely wrote loose functions like df['rolling_mean'] =
df['val'].rolling(5).mean().﻿
 * In the IDE, create
src/features/telemetry_pipeline.py. Encapsulate transformations into a class
that handles both batched training DataFrames and single JSON payloads at
inference:﻿
# src/features/telemetry_pipeline.py﻿
import numpy as np﻿
import scipy.fft﻿
 
class TelemetryFeatureExtractor:﻿
    def __init__(self, window_sizes=(5,
15, 60)):﻿
        self.window_sizes
= window_sizes﻿
 
    def extract_fft_energy(self, series:
np.ndarray) -> float:﻿
        if
len(series) < 8:﻿
            return
0.0﻿
        fft_vals =
scipy.fft.fft(series)﻿
        return
float(np.sum(np.abs(fft_vals) ** 2) / len(series))﻿
 
    def transform_stream(self,
recent_history: np.ndarray) -> dict:﻿
        """Transforms
a live sliding window of metric floats into ML features."""﻿
        features =
{}﻿
        for w in
self.window_sizes:﻿
            slice_w
= recent_history[-w:] if len(recent_history) >= w else recent_history﻿
            features[f"mean_{w}"]
= float(np.mean(slice_w))﻿
            features[f"std_{w}"]
= float(np.std(slice_w))﻿
        features["fft_energy"]
= self.extract_fft_energy(recent_history)﻿
        return
features﻿
 
2. Save Serialized Model Weights to Disk﻿
Do not bundle training data into your production app. Save
only lightweight inference artifacts:﻿
 * LightGBM:
joblib.dump(model, "models/lightgbm_anomaly.pkl")﻿
 * SLM LoRA Adapter:
model.save_pretrained("models/slm_lora_adapter")﻿
 * Quantized SLM:
Export to GGUF or INT8 ONNX (models/slm_quantized.onnx).﻿
What Happens in the IDE (VS Code)﻿
1. Modular Model Loaders (src/models/inference.py)﻿
Write an inference engine that boots models once during
startup and keeps them cached in memory:﻿
# src/models/inference.py﻿
import joblib﻿
import numpy as np﻿
import onnxruntime as ort﻿
 
class HybridInferenceEngine:﻿
    def __init__(self,
tabular_model_path: str, slm_onnx_path: str):﻿
        # Load
tabular model﻿
        self.tabular_model
= joblib.load(tabular_model_path)﻿
        # Load INT8
Quantized SLM into ONNX Runtime (optimized for x86 CPU AVX2)﻿
        self.slm_session
= ort.InferenceSession(﻿
            slm_onnx_path,﻿
            providers=["CPUExecutionProvider"]﻿
        )﻿
 
    def predict_anomaly(self,
feature_vector: np.ndarray) -> float:﻿
        # Returns
failure probability (sub-5ms)﻿
        return
float(self.tabular_model.predict_proba(feature_vector)[0][1])﻿
 
    def parse_log(self, log_text: str)
-> dict:﻿
        # Runs
tokenized log through SLM session & extracts structured JSON﻿
        # ...
tokenization -> ONNX run -> decode output ...﻿
        return
{"error_domain": "DiskIO", "severity":
"CRITICAL", "action": "Clear cache"}﻿
 
2. Fusion Logic (src/models/fusion.py)﻿
Combine tabular and SLM predictions into a unified triage
rule:﻿
def compute_incident_triage(anomaly_prob: float,
slm_analysis: dict) -> dict:﻿
    severity_weight = 1.0 if
slm_analysis["severity"] == "CRITICAL" else 0.5﻿
    fused_score = (0.6 * anomaly_prob) +
(0.4 * severity_weight)﻿
﻿
    return {﻿
        "triage_score":
round(fused_score, 3),﻿
        "is_p1_incident":
fused_score >= 0.75,﻿
        "root_cause":
slm_analysis.get("probable_cause", "Unknown"),﻿
        "routing_group":
"Platform-SRE" if "Disk" in
slm_analysis.get("error_domain", "") else
"App-Dev"﻿
    }﻿
 
3. API Service (src/app/main.py)﻿
Expose the system via FastAPI using Pydantic schemas:﻿
# src/app/main.py﻿
from fastapi import FastAPI﻿
from pydantic import BaseModel﻿
import numpy as np﻿
from src.features.telemetry_pipeline import
TelemetryFeatureExtractor﻿
from src.models.inference import HybridInferenceEngine﻿
from src.models.fusion import compute_incident_triage﻿
 
app = FastAPI(title="IT Incident Triage Engine")﻿
 
# Preload pipeline and models at server boot﻿
extractor = TelemetryFeatureExtractor()﻿
engine =
HybridInferenceEngine("models/lightgbm_anomaly.pkl",
"models/slm_quantized.onnx")﻿
 
class IncidentPayload(BaseModel):﻿
    log_line: str﻿
    telemetry_window: list[float]﻿
 
@app.post("/api/v1/triage")﻿
def triage_incident(payload: IncidentPayload):﻿
    # 1. Feature extraction from raw
metrics﻿
    features =
extractor.transform_stream(np.array(payload.telemetry_window))﻿
    feat_array =
np.array([list(features.values())])﻿
﻿
    # 2. Parallel model inference﻿
    p_anomaly =
engine.predict_anomaly(feat_array)﻿
    log_insights =
engine.parse_log(payload.log_line)﻿
﻿
    # 3. Multimodal Decision Fusion﻿
    decision =
compute_incident_triage(p_anomaly, log_insights)﻿
    return decision﻿
 
4. Containerization (Dockerfile)﻿
Containerize the app with multi-threading limits matching
your hardware:﻿
FROM python:3.11-slim﻿
WORKDIR /app﻿
 
COPY requirements.txt .﻿
RUN pip install --no-cache-dir -r requirements.txt﻿
 
COPY src/ ./src/﻿
COPY models/ ./models/﻿
 
# Match thread pool to your CPU physical cores for low
latency﻿
ENV OMP_NUM_THREADS=4﻿
ENV MKL_NUM_THREADS=4﻿
 
EXPOSE 8000﻿
CMD ["uvicorn", "src.app.main.py:app",
"--host", "0.0.0.0", "--port", "8000"]

