# MLOps Batch Job — Primetrade.ai Internship Assessment

A minimal MLOps-style batch pipeline that loads OHLCV market data, computes a rolling-mean trading signal, and outputs structured metrics — fully Dockerized for one-command execution.

---

## Project Structure

```
primetradeML/
├── run.py              # Main pipeline script
├── config.yaml         # Configuration (seed, window, version)
├── data.csv            # Input OHLCV dataset (10,000 rows)
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker build definition
├── README.md           # This file
├── metrics.json        # Sample output (generated after run)
└── run.log             # Sample log (generated after run)
```

---

## Prerequisites

- Python 3.9+
- pip
- Docker Desktop

---

## Local Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the pipeline
```bash
python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log
```

### 3. View outputs
```bash
# Metrics
cat metrics.json

# Logs
cat run.log
```

---

## Docker Build & Run

```bash
# Build the image (pipeline runs during build)
docker build -t mlops-task .

# Run the container (prints metrics.json to stdout)
docker run --rm mlops-task
```

Expected exit code: `0` on success, non-zero on failure.

---

## Config (`config.yaml`)

```yaml
seed: 42       # Random seed for reproducibility
window: 5      # Rolling mean window size
version: "v1"  # Pipeline version tag
```

---

## Example `metrics.json` Output

```json
{
  "version": "v1",
  "rows_processed": 10000,
  "metric": "signal_rate",
  "value": 0.499,
  "latency_ms": 85,
  "seed": 42,
  "status": "success"
}
```

---

## Error Output Format

If the pipeline fails (missing file, bad config, etc.), `metrics.json` will contain:

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Description of what went wrong"
}
```

---

## Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| NaN handling | First `window-1` rows excluded from signal | Consistent, no arbitrary fill |
| `rows_processed` | Total rows in CSV (including NaN window rows) | Reflects actual data loaded |
| `signal_rate` | Mean over valid (non-NaN) signal rows only | Accurate representation |
| Seed | `numpy.random.seed(seed)` from config | Fully reproducible |
| Error on any failure | Writes error JSON + exits non-zero | Ensures metrics.json always exists |

---

## Validation & Error Handling

The pipeline validates:
- Config file exists and has all required fields (`seed`, `window`, `version`)
- Input CSV exists, is non-empty, and has a `close` column
- `close` values are numeric
- Any unexpected runtime exception writes an error metrics file before exiting

---

*Built for Primetrade.ai ML/MLOps Engineering Internship Technical Assessment*
