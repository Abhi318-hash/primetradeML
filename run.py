"""
run.py — MLOps Batch Job Pipeline
Primetrade.ai ML/MLOps Engineering Internship — Technical Assessment

Usage:
    python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ─────────────────────────────────────────────
# Argument Parsing
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="MLOps batch job: rolling-mean signal pipeline."
    )
    parser.add_argument("--input",    required=True, help="Path to input data CSV")
    parser.add_argument("--config",   required=True, help="Path to YAML config file")
    parser.add_argument("--output",   required=True, help="Path to write metrics JSON")
    parser.add_argument("--log-file", required=True, help="Path to write log file")
    return parser.parse_args()


# ─────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────
def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("mlops_pipeline")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler — detailed logs
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler — info and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ─────────────────────────────────────────────
# Metrics Writer
# ─────────────────────────────────────────────
def write_metrics(output_path: str, payload: dict, logger: logging.Logger):
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Metrics written to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to write metrics file: {e}")


# ─────────────────────────────────────────────
# Step 1 — Load & Validate Config
# ─────────────────────────────────────────────
def load_config(config_path: str, logger: logging.Logger) -> dict:
    logger.info(f"Loading config from: {config_path}")

    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Config file is empty or not valid YAML.")

    required_fields = {"seed", "window", "version"}
    missing = required_fields - config.keys()
    if missing:
        raise ValueError(f"Config missing required fields: {missing}")

    if not isinstance(config["seed"], int):
        raise ValueError(f"Config 'seed' must be an integer, got: {type(config['seed'])}")
    if not isinstance(config["window"], int) or config["window"] < 1:
        raise ValueError(f"Config 'window' must be a positive integer, got: {config['window']}")
    if not isinstance(config["version"], str):
        raise ValueError(f"Config 'version' must be a string, got: {type(config['version'])}")

    logger.info(
        f"Config loaded — seed={config['seed']}, window={config['window']}, version={config['version']}"
    )
    return config


# ─────────────────────────────────────────────
# Step 2 — Load & Validate Dataset
# ─────────────────────────────────────────────
def load_dataset(input_path: str, logger: logging.Logger) -> pd.DataFrame:
    logger.info(f"Loading dataset from: {input_path}")

    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        raise ValueError(f"Failed to parse CSV file: {e}")

    if df.empty:
        raise ValueError("Input CSV is empty — no rows found.")

    if "close" not in df.columns:
        raise ValueError(
            f"Required column 'close' not found. Available columns: {list(df.columns)}"
        )

    # Coerce close to numeric, catch non-numeric data
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    null_count = df["close"].isna().sum()
    if null_count == len(df):
        raise ValueError("All values in 'close' column are non-numeric / NaN.")

    if null_count > 0:
        logger.warning(f"{null_count} non-numeric values found in 'close' — they will be treated as NaN.")

    logger.info(f"Dataset loaded — {len(df)} rows, {len(df.columns)} columns: {list(df.columns)}")
    return df


# ─────────────────────────────────────────────
# Step 3 — Rolling Mean
# ─────────────────────────────────────────────
def compute_rolling_mean(df: pd.DataFrame, window: int, logger: logging.Logger) -> pd.Series:
    logger.info(f"Computing rolling mean on 'close' with window={window}")
    rolling_mean = df["close"].rolling(window=window, min_periods=window).mean()
    nan_count = rolling_mean.isna().sum()
    logger.debug(
        f"Rolling mean computed — {nan_count} NaN rows (first window-1={window - 1} rows, excluded from signal)"
    )
    return rolling_mean


# ─────────────────────────────────────────────
# Step 4 — Signal Generation
# ─────────────────────────────────────────────
def compute_signal(df: pd.DataFrame, rolling_mean: pd.Series, logger: logging.Logger) -> pd.Series:
    logger.info("Generating binary signal: 1 if close > rolling_mean, else 0")

    # Only compute signal where rolling_mean is valid (non-NaN)
    valid_mask = rolling_mean.notna()
    signal = pd.Series(np.nan, index=df.index)
    signal[valid_mask] = (df.loc[valid_mask, "close"] > rolling_mean[valid_mask]).astype(int)

    valid_signals = signal[valid_mask]
    logger.info(
        f"Signal generated — {valid_mask.sum()} valid rows, "
        f"{int(valid_signals.sum())} signal=1, {int((valid_signals == 0).sum())} signal=0"
    )
    return signal


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    args = parse_args()

    # ── Logging first (we need it for error reporting) ──────────────────
    logger = setup_logging(args.log_file)
    logger.info("=" * 60)
    logger.info("MLOps Batch Job — STARTED")
    logger.info("=" * 60)
    logger.info(f"Arguments: input={args.input}, config={args.config}, output={args.output}, log={args.log_file}")

    start_time = time.perf_counter()
    version = "unknown"  # fallback for error metrics

    try:
        # ── Step 1: Config ───────────────────────────────────────────────
        config = load_config(args.config, logger)
        version = config["version"]
        seed    = config["seed"]
        window  = config["window"]

        # Set random seed for reproducibility
        np.random.seed(seed)
        logger.info(f"Random seed set: numpy.random.seed({seed})")

        # ── Step 2: Dataset ──────────────────────────────────────────────
        df = load_dataset(args.input, logger)
        rows_processed = len(df)

        # ── Step 3: Rolling Mean ─────────────────────────────────────────
        rolling_mean = compute_rolling_mean(df, window, logger)

        # ── Step 4: Signal ───────────────────────────────────────────────
        signal = compute_signal(df, rolling_mean, logger)

        # ── Step 5: Metrics ──────────────────────────────────────────────
        valid_signal = signal.dropna()
        signal_rate  = float(round(valid_signal.mean(), 4))

        elapsed_ms = round((time.perf_counter() - start_time) * 1000)

        metrics = {
            "version":        version,
            "rows_processed": rows_processed,
            "metric":         "signal_rate",
            "value":          signal_rate,
            "latency_ms":     elapsed_ms,
            "seed":           seed,
            "status":         "success",
        }

        logger.info("-" * 60)
        logger.info("METRICS SUMMARY")
        logger.info(f"  rows_processed : {rows_processed}")
        logger.info(f"  signal_rate    : {signal_rate}")
        logger.info(f"  latency_ms     : {elapsed_ms}")
        logger.info(f"  version        : {version}")
        logger.info(f"  seed           : {seed}")
        logger.info("-" * 60)

        write_metrics(args.output, metrics, logger)

        logger.info("MLOps Batch Job — COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

        # Print final metrics to stdout (required by Docker CMD)
        print(json.dumps(metrics, indent=2))
        sys.exit(0)

    except Exception as e:
        elapsed_ms = round((time.perf_counter() - start_time) * 1000)
        error_msg  = str(e)

        logger.error(f"PIPELINE FAILED: {error_msg}", exc_info=True)
        logger.info("Writing error metrics...")

        error_metrics = {
            "version":       version,
            "status":        "error",
            "error_message": error_msg,
        }

        write_metrics(args.output, error_metrics, logger)

        logger.info("MLOps Batch Job — FAILED")
        logger.info("=" * 60)

        print(json.dumps(error_metrics, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
