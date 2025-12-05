import pandas as pd
import numpy as np

# --- Load sensor data ---
def load_sensor_data(csv_path="/Users/mac/Documents/Python Projects/Agentic_Ai/qc_simulation/sensor_logs/sensor_logs.csv"):
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        raise FileNotFoundError(f"Could not load sensor log: {e}")

# --- Compute metrics ---
def compute_metrics(df, confidence_threshold, manual_band, vibration_weight, acoustic_weight, sampling_rate):
    """
    df: dataframe with columns [part_id, vibration_rms, acoustic_db, temp_c, label]
    confidence_threshold: float (0-1)
    manual_band: tuple (low, high)
    vibration_weight, acoustic_weight: floats normalized to sum=1
    sampling_rate: float (0-1)
    """

    # Confidence score = weighted vibration + weighted acoustic
    df["confidence"] = (
        vibration_weight * (df["vibration_rms"] / df["vibration_rms"].max()) +
        acoustic_weight * (df["acoustic_db"] / df["acoustic_db"].max())
    )

    # AI decision
    df["ai_decision"] = np.where(df["confidence"] >= confidence_threshold, "DEFECT", "OK")

    # Manual override zone
    df["manual_check"] = df["confidence"].between(manual_band[0], manual_band[1])

    # Ground truth
    y_true = df["label"].values
    y_pred = df["ai_decision"].values

    # Metrics
    tp = np.sum((y_true == "DEFECT") & (y_pred == "DEFECT"))
    tn = np.sum((y_true == "OK") & (y_pred == "OK"))
    fp = np.sum((y_true == "OK") & (y_pred == "DEFECT"))
    fn = np.sum((y_true == "DEFECT") & (y_pred == "OK"))

    accuracy = (tp + tn) / len(df)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Throughput model
    base_takt = 45  # seconds per part baseline
    manual_penalty = 10 * df["manual_check"].mean()  # avg extra time per manual check
    sampling_penalty = 5 * sampling_rate
    takt_time = base_takt + manual_penalty + sampling_penalty
    jobs_per_hour = 3600 / takt_time

    # Costs (simple model)
    inspection_cost = 1000 * df["manual_check"].mean() + 500 * sampling_rate
    defect_cost = fn * 100  # each missed defect costs $100

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
