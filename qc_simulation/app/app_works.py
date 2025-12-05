import streamlit as st
import plotly.express as px
from Agentic_AI.qc_simulation.app.backend_works import load_sensor_data, compute_metrics

st.title("AI-Enabled Quality Control Simulator")

# --- Sidebar controls ---
st.sidebar.header("Workflow Settings")

# --- Sliders first ---
confidence_threshold = st.sidebar.slider(
    "Confidence threshold", 0.0, 1.0, 0.7, 0.01,
    help="Minimum confidence score required for AI to auto‑accept a part."
)

manual_band = st.sidebar.slider(
    "Manual verification band", 0.0, 1.0, (0.5, 0.69), 0.01,
    help="Range of confidence scores where human inspectors double‑check AI decisions."
)

vibration_weight = st.sidebar.slider(
    "Vibration weight", 0.0, 1.0, 0.6, 0.05,
    help="Relative importance of vibration data in defect confidence scoring."
)

acoustic_weight = st.sidebar.slider(
    "Acoustic weight", 0.0, 1.0, 0.4, 0.05,
    help="Relative importance of acoustic (sound) data in defect confidence scoring."
)

# Normalize weights
total = vibration_weight + acoustic_weight
if total > 0:
    vibration_weight /= total
    acoustic_weight /= total

sampling_rate = st.sidebar.slider(
    "End-of-line sampling rate (%)", 0, 100, 10, 5,
    help="Percentage of parts randomly inspected at the end of the line."
) / 100.0

st.sidebar.caption("💡 Deploy = making this workflow available on the factory line or hosting it online.")

# --- Scenario Presets + Reset under sliders ---
st.sidebar.subheader("Scenario Presets")

row1_col1, row1_col2 = st.sidebar.columns(2)
row2_col1, row2_col2 = st.sidebar.columns(2)

if row1_col1.button("Conservative"):
    confidence_threshold = 0.65
    manual_band = (0.50, 0.69)
    vibration_weight, acoustic_weight = 0.7, 0.3
    sampling_rate = 0.20

if row1_col2.button("Aggressive"):
    confidence_threshold = 0.85
    manual_band = (0.60, 0.65)
    vibration_weight, acoustic_weight = 0.5, 0.5
    sampling_rate = 0.05

if row2_col1.button("Balanced"):
    confidence_threshold = 0.75
    manual_band = (0.55, 0.68)
    vibration_weight, acoustic_weight = 0.6, 0.4
    sampling_rate = 0.10

if row2_col2.button("Reset"):
    confidence_threshold = 0.7
    manual_band = (0.5, 0.69)
    vibration_weight, acoustic_weight = 0.6, 0.4
    sampling_rate = 0.10

# --- Load data ---
df = load_sensor_data()

# --- Compute metrics ---
results = compute_metrics(df, confidence_threshold, manual_band,
                          vibration_weight, acoustic_weight, sampling_rate)

# --- Display KPIs ---
st.subheader("Performance Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy (%)", results["accuracy"], help="Overall correctness of AI + manual workflow.")
col2.metric("Recall (%)", results["recall"], help="Percentage of defects correctly identified.")
col3.metric("Precision (%)", results["precision"], help="Percentage of flagged parts that were truly defective.")

col4, col5, col6 = st.columns(3)
col4.metric("False Negatives", results["false_negatives"], help="Defects missed and shipped to customers.")
col5.metric("Takt Time (sec)", results["takt_time"], help="Average seconds per part.")
col6.metric("Jobs per Hour", results["jobs_per_hour"], help="Throughput rate of the line.")

col7, col8 = st.columns(2)
col7.metric("Inspection Cost ($)", results["inspection_cost"], help="Estimated cost of inspections.")
col8.metric("Defect Cost ($)", results["defect_cost"], help="Estimated cost of missed defects.")

# --- Scatter plot ---
st.subheader("Confidence Distribution")
fig = px.scatter(results["df"], x="part_id", y="confidence", color="label",
                 title="Confidence by Part", labels={"confidence": "Confidence Score"})
st.plotly_chart(fig, use_container_width=True)

# --- Sample image (optional) ---
st.subheader("Sample Part Image")
st.caption("Example of what the AI sees during inspection.")
st.image("../images/001.png", caption="Sample part", use_column_width=True)
