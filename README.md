# Real-Time IT Incident Triage Engine

This repository packages the prototype work into a reproducible incident triage system:

- Telemetry anomaly detection for the CPU utilization series in `data/cpu_utilization_asg_misconfiguration.csv`
- Log root-cause extraction for the structured HDFS sample in `data/HDFS_2k.log_structured.csv`
- A fusion layer that combines both signals into a single triage decision
- A FastAPI service for online inference
- Training and benchmark scripts for the tabular and log paths

The notebooks remain the experimental sandbox:

- `notebooks/01_signall_and_tabular_proto.ipynb`
- `notebooks/02_slm_lora_ssanity_check.ipynb`

## What is implemented

- Reusable telemetry feature extraction
- Time-aware telemetry labeling for the provided NAB-style CPU dataset
- A calibrated anomaly model with LightGBM when available, plus a scikit-learn fallback
- HDFS log heuristic labeling, prompt formatting, and JSON validation
- A fusion layer that turns telemetry risk plus log severity into a priority and route
- FastAPI endpoint `/api/v1/triage`
- Unit-testable core logic and artifact helpers

## Repo layout

```text
incident-triage-engine/
├── data/
├── incident_triage/
├── notebooks/
├── tests/
├── artifacts/           # created at runtime
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

If you only want the tabular pipeline, the optional LoRA packages can be skipped.

## Train the telemetry model

```bash
python -m incident_triage.train_tabular \
  --input data/cpu_utilization_asg_misconfiguration.csv \
  --output-dir artifacts
```

This will:

- load and label the telemetry series
- build rolling, EWMA, and spectral features
- train and calibrate the anomaly model
- save the model, feature list, and metrics into `artifacts/`

## Prepare the log SLM dataset

```bash
python -m incident_triage.train_slm \
  --input data/HDFS_2k.log_structured.csv \
  --output-dir artifacts \
  --dry-run
```

This exports a prompt/response JSONL dataset. If you want to actually run LoRA fine-tuning, pass `--train`.

## Run the API

```bash
uvicorn incident_triage.api.main:app --reload
```

Example request body:

```json
{
  "telemetry": {
    "values": [32.0, 31.5, 32.1, 33.2, 33.8, 35.1, 36.0]
  },
  "log_content": "PacketResponder 2 for block blk_902318412891 terminating due to socket exception"
}
```

## API response

The service returns:

- telemetry anomaly risk
- log analysis JSON
- fused incident score
- priority
- routing recommendation

## Notes

- The telemetry model is configured for the CSV already present in `data/`
- The log path uses heuristic labeling and validation for the HDFS sample
- The SLM training path is written to be CPU-friendly, but the repository ships without a pretrained adapter

