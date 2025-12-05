import pandas as pd
import numpy as np
import os
from datetime import datetime

# --- Load sensor data ---
def load_sensor_data(csv_path="../sensor_logs/sensor_logs.csv"):
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        raise FileNotFoundError(f"Could not load sensor log: {e}")

# --- Compute metrics ---
def compute_metrics(df, confidence_threshold, manual_band, vibration_weight, acoustic_weight, sampling_rate):
    df["confidence"] = (
        vibration_weight * (df["vibration_rms"] / df["vibration_rms"].max()) +
        acoustic_weight * (df["acoustic_db"] / df["acoustic_db"].max())
    )

    df["ai_decision"] = np.where(df["confidence"] >= confidence_threshold, "DEFECT", "OK")
    df["manual_check"] = df["confidence"].between(manual_band[0], manual_band[1])

    y_true = df["label"].values
    y_pred = df["ai_decision"].values

    tp = np.sum((y_true == "DEFECT") & (y_pred == "DEFECT"))
    tn = np.sum((y_true == "OK") & (y_pred == "OK"))
    fp = np.sum((y_true == "OK") & (y_pred == "DEFECT"))
    fn = np.sum((y_true == "DEFECT") & (y_pred == "OK"))

    accuracy = (tp + tn) / len(df)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    base_takt = 45
    manual_penalty = 10 * df["manual_check"].mean()
    sampling_penalty = 5 * sampling_rate
    takt_time = base_takt + manual_penalty + sampling_penalty
    jobs_per_hour = 3600 / takt_time

    inspection_cost = 1000 * df["manual_check"].mean() + 500 * sampling_rate
    defect_cost = fn * 100

    return {
        "accuracy": round(accuracy * 100, 1),
        "recall": round(recall * 100, 1),
        "precision": round(precision * 100, 1),
        "false_negatives": int(fn),
        "takt_time": round(takt_time, 1),
        "jobs_per_hour": round(jobs_per_hour, 1),
        "inspection_cost": round(inspection_cost, 1),
        "defect_cost": round(defect_cost, 1),
        "df": df
    }

# --- Participant logging ---
def log_run(team_name, confidence_threshold, manual_band, vibration_weight, acoustic_weight, sampling_rate, results, log_path="../logs/participant_runs.csv"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "team": team_name,
        "confidence_threshold": confidence_threshold,
        "manual_band_low": manual_band[0],
        "manual_band_high": manual_band[1],
        "vibration_weight": vibration_weight,
        "acoustic_weight": acoustic_weight,
        "sampling_rate": sampling_rate,
        "accuracy": results["accuracy"],
        "recall": results["recall"],
        "precision": results["precision"],
        "false_negatives": results["false_negatives"],
        "takt_time": results["takt_time"],
        "jobs_per_hour": results["jobs_per_hour"],
        "inspection_cost": results["inspection_cost"],
        "defect_cost": results["defect_cost"]
    }

    df_log = pd.DataFrame([log_entry])
    if os.path.exists(log_path):
        df_log.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df_log.to_csv(log_path, mode="w", header=True, index=False)

# --- View logged runs ---
def view_logs(log_path="../logs/participant_runs.csv"):
    if os.path.exists(log_path):
        return pd.read_csv(log_path)
    else:
        return pd.DataFrame()

# --- ROI calculation ---
def compute_roi(baseline_defect_cost, current_defect_cost, inspection_cost, investment_cost):
    """
    ROI = ((Baseline defect cost - Current defect cost) - Inspection cost) / Investment cost
    Returns None if investment_cost is zero or negative.
    """
    if investment_cost is None or investment_cost <= 0:
        return None
    savings = (baseline_defect_cost - current_defect_cost) - inspection_cost
    return round(savings / investment_cost, 3)

# --- Helper: lookup original label by part_id ---
def lookup_part_label(df, part_id):
    try:
        pid = int(part_id)
    except Exception:
        return None
    row = df[df["part_id"] == pid]
    if row.empty:
        return None
    return row["label"].values[0]
