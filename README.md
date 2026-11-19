<div align="center">

# ⚡ Real-Time IT Incident Triage Engine

### *Multimodal AI System for Automated Anomaly Detection, Root-Cause Analysis, & Incident Routing*

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LightGBM](https://img.shields.io/badge/LightGBM-Calibrated-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![Redis](https://img.shields.io/badge/Redis-Distributed_State-DC382D.svg?logo=redis&logoColor=white)](https://redis.io)
[![Tests](https://img.shields.io/badge/Tests-10%20Passed-success.svg)](./tests)
[![Data Leakage](https://img.shields.io/badge/Temporal_Split-Zero_Leakage-purple.svg)](#statistical-rigor--zero-leakage-guarantee)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

<p align="center">
  <b>Sub-100ms automated incident triage fusing continuous time-series telemetry with unstructured distributed logs.</b><br>
  Built for SRE and Platform Engineering teams to crush MTTA & MTTD without alert fatigue.
</p>

---

</div>

## 📌 Executive Overview

In cloud-native architectures, on-call engineers face two critical monitoring bottlenecks:
1. **Metric Silos (Prometheus/Datadog)** detect numerical spikes (e.g. CPU > 85%) but lack causal context (is it a routine batch job or a thread deadlock?).
2. **Log Silos (Splunk/Elasticsearch)** flood engineers with noisy error messages (`ERROR`, `FATAL`) but lack overall infrastructure health context.

The **Real-Time IT Incident Triage Engine** solves this by unifying both modalities into an online inference pipeline:

```mermaid
flowchart TD
    subgraph Ingest ["1. Ingestion Layer"]
        M["Point-by-Point Metric Stream\n(OpenTelemetry / Datadog Agent)"] --> B["StreamingWindowBuffer\n(Atomic Redis / In-Memory Fallback)"]
        L["Unstructured Log Stream\n(FluentBit / Vector)"] --> Sec["DoS Protection Guardrail\n(Payload Ceiling: 10KB)"]
    end

    subgraph Analytics ["2. Applied ML & NLP Analytics"]
        B --> FE["Feature Extractor\n(Rolling Moments, EWMA, Diff, Spread)"]
        FE --> Model["Calibrated LightGBM Model\n(Temporal Split, Prefit Sigmoid)"]
        Model --> TR["Calibrated Telemetry Risk\nP(Anomaly | Features)"]

        Sec --> SLM["Log Root-Cause Engine\n(Guided Decoding + Auto-Repair)"]
        SLM --> LA["Structured Log Analysis\n(Severity, Domain, Probable Cause)"]
    end

    subgraph Decision ["3. Signal Fusion & Actionable Routing"]
        TR --> Core["Multimodal Fusion Core\n(Weighting + Synergy Bonus)"]
        LA --> Core
        Core --> Out["Actionable Triage Decision\n(P1 / P2 / P3 & Subsystem Route)"]
    end

    subgraph Drift ["4. Continuous Observability"]
        B --> PSI["Population Stability Index (PSI)\nQuantile Baseline Divergence"]
        PSI --> Health["GET /health/drift\n(HEALTHY / MODERATE / CRITICAL)"]
    end
```

---

## 🚀 Key Features

- **Multimodal Signal Fusion**: Evaluates continuous metric trajectories alongside unstructured log text, applying a **synergy bonus** when high-risk anomalies coincide with critical error traces.
- **Statistical Rigor (Zero Data Leakage)**: Strict **60% Train $\to$ 20% Tune $\to$ 20% Holdout** sequential temporal split. Features are calibrated via Platt Scaling (`cv="prefit"`), outputting true posterior probabilities.
- **Horizontal Scaling & Stateful Streaming**: Built-in `RedisStreamingBuffer` uses atomic `LPUSH` + `LTRIM` operations to maintain rolling windows across multi-worker ASGI (`uvicorn -w 4`) and distributed Kubernetes pods.
- **3-Tier Zero-Crash Log Shield**: Guided JSON generation $\to$ strict schema auto-repair normalizer $\to$ deterministic heuristic fallback. Guarantees **0% unhandled 500 API exceptions**.
- **Population Stability Index (PSI) Drift Health**: Quantile divergence monitoring continuously tracks concept drift between baseline training telemetry and live incoming production streams.
- **Production Hardened**: Header authentication (`X-API-Key`), 10KB DoS payload limits, and Docker/Kubernetes healthcheck probes.

---

## 📊 Empirical Generalization Benchmarks

Trained on real AWS EC2 Auto Scaling Group misconfiguration series (NAB benchmark) with zero temporal leakage:

| Metric Partition | Sample Count | Class 0 (Normal) | Class 1 (Anomaly) | Overall / Macro |
| :--- | :---: | :---: | :---: | :---: |
| **Train Window** | 10,830 rows | Base Estimator Fit | Base Estimator Fit | — |
| **Tune Window** | 3,610 rows | Sigmoid Calibration | Optimal Threshold ($t^*=0.05$) | — |
| **Test Holdout** | 3,610 rows | **$F_1$: 0.906** (Recall: 1.00) | **$F_1$: 0.359** (Precision: 1.00) | **83.57% Accuracy** |

> **Key Takeaway**: When the model flags an anomaly on holdout test data, **precision is 100%**. Zero false-positive pages on holdout telemetry.

---

## ⚡ Quickstart

### 1. Clone & Setup Environment
```bash
git clone https://github.com/your-org/incident-triage-engine.git
cd incident-triage-engine

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Verify Pre-Trained Model Artifacts
Model artifacts are already pre-trained and saved in `artifacts/`:
- `artifacts/tabular_model.joblib` (Calibrated LightGBM model)
- `artifacts/tabular_metrics.json` (Holdout evaluation metrics)
- `artifacts/tabular_feature_columns.json` (Ordered input features)

*(Optional) Re-train anytime via:*
```bash
python -m incident_triage.train_tabular \
  --input data/cpu_utilization_asg_misconfiguration.csv \
  --output-dir artifacts
```

### 3. Launch the API Service
```bash
uvicorn incident_triage.api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📡 API Usage & Endpoints

### A. Real-Time Streaming Triage (`POST /api/v1/triage`)
Ingest streaming single-point metrics from monitoring agents or full historical batches:

```bash
curl -X POST http://localhost:8000/api/v1/triage \
  -H "Content-Type: application/json" \
  -d '{
    "telemetry": {
      "value": 84.5
    },
    "asset_id": "us-east-1-prod-worker-08",
    "log_content": "PacketResponder 2 for block blk_902318412891 terminating due to socket exception"
  }'
```

#### Response (`200 OK` in < 25ms):
```json
{
  "incident_score": 0.93,
  "priority": "P1",
  "route_to": "incident-bridge",
  "explanation": "Telemetry risk 0.89, log severity CRITICAL with confidence 1.00 produced fused score 0.93.",
  "telemetry": {
    "risk_score": 0.89,
    "threshold": 0.05,
    "is_anomaly": true,
    "probability": 0.89,
    "feature_snapshot": {
      "rolling_mean_5": 81.2,
      "ewma_0.1": 79.4,
      "diff_1": 3.1
    }
  },
  "log_analysis": {
    "error_domain": "HDFS_DataNode",
    "severity": "CRITICAL",
    "probable_cause": "Packet responder termination, timeout, or storage subsystem failure",
    "confidence": 1.0,
    "raw_text": "PacketResponder 2 for block blk_902318412891 terminating due to socket exception"
  }
}
```

---

### B. Health & Cluster Status (`GET /health`)
```bash
curl -f http://localhost:8000/health
```
```json
{
  "status": "ok",
  "telemetry_model": "loaded",
  "buffer_backend": "InMemoryStreamingBuffer"
}
```
*(When `REDIS_URL` is set, `buffer_backend` automatically activates `RedisStreamingBuffer`).*

---

### C. Concept Drift Telemetry (`GET /health/drift`)
```bash
curl -f http://localhost:8000/health/drift
```
```json
{
  "status": "HEALTHY",
  "psi": 0.0384,
  "drift_detected": false,
  "sample_count": 30
}
```
- **$\text{PSI} < 0.10$**: `HEALTHY` (Baseline stable).
- **$0.10 \le \text{PSI} < 0.25$**: `MODERATE_DRIFT` (Warning to ML team).
- **$\text{PSI} \ge 0.25$**: `CRITICAL_DRIFT` (Automated alert to retrain model).

---

## 🔒 Security & Authentication

1. **Header Authentication**: Set `API_KEY="your-secret-key"` in the environment to enforce `X-API-Key: your-secret-key` on all triage endpoints. If unset, it gracefully operates in local development mode.
2. **Payload Protection**: `log_content` payloads exceeding 10,000 characters are automatically rejected with `422 Unprocessable Entity` to eliminate memory exhaustion DoS attacks.

---

## 🐳 Production Deployment

### Docker Multi-Worker Run
```bash
docker build -t incident-triage:latest .
docker run -d \
  -p 8000:8000 \
  -e API_KEY="prod-secret-key" \
  -e REDIS_URL="redis://redis.internal:6379/0" \
  --name incident-triage \
  incident-triage:latest
```

### Kubernetes Deployment Snippet
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: incident-triage
spec:
  replicas: 4
  template:
    spec:
      containers:
      - name: triage-engine
        image: incident-triage:latest
        ports:
        - containerPort: 8000
        env:
        - name: API_KEY
          valueFrom:
            secretKeyRef:
              name: triage-secrets
              key: api-key
        - name: REDIS_URL
          value: "redis://redis-service:6379/0"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
```

---

## 🧪 Test Suite Execution

Run the complete 10-test unit and production-hardening suite:
```bash
python -m pytest tests/test_fusion.py tests/test_logs.py tests/test_telemetry.py tests/test_production_hardening.py -v
```
```text
======================= 10 passed in 4.34s =======================
```

---

## 📖 Deep-Dive Architectural Documentation

For full mathematical derivations of Platt probability calibration, Population Stability Index (PSI) formulation, LoRA distillation parameters, and multimodal fusion theorems, read the [Master Architecture Manual (MASTER_DOCUMENTATION.md)](./MASTER_DOCUMENTATION.md).

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
