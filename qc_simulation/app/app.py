import os
import streamlit as st
import plotly.express as px
from backend import (
    load_sensor_data,
    compute_metrics,
    log_run,
    view_logs,
    compute_roi,
    lookup_part_label,
)

st.title("AI-Enabled Quality Control Simulator (Automotive)")

# --- Scenario & Instructions ---
with st.expander("ℹ️ Scenario & Instructions"):
    st.markdown("""
    ### Scenario: AI-Enabled Quality Control in Automotive Manufacturing
    Your factory is part of the **automotive sector**, producing precision parts that must meet strict safety and quality standards before being assembled into vehicles. Currently, the factory relies on a **manual inspection process**: human inspectors visually check each part as it comes down the line every **45 seconds**. This process is slow, inconsistent, and prone to human error. The inspection logs and defect images from this manual process form the baseline dataset you will work with.

    Management is considering an **AI-enabled quality control (QC) system** that uses sensor data and computer vision to detect defects in real time. The goal is to improve accuracy, reduce inspection delays, and optimize costs while maintaining throughput.

    ### Instructions to Participants
    - **Roles:** Production Manager, Quality Engineer, Data Analyst, Operations Lead.
    - **Steps:** Briefing → Data Review → Workflow Design (thresholds, inspection points, escalation rules) → Presentation → Scoring & Debrief.
    - **Scoring Criteria:** Accuracy (30%), Efficiency (30%), Innovation (20%), Collaboration (20%).
    - **Key Considerations:** Higher accuracy may slow throughput; lower inspection frequency may reduce costs but risk escapes; ROI depends on balancing defect reduction with AI investment.
    """)

# --- Sidebar controls ---
st.sidebar.header("Workflow Settings")

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

# Normalize weights to sum to 1
total = vibration_weight + acoustic_weight
if total > 0:
    vibration_weight /= total
    acoustic_weight /= total

sampling_rate = st.sidebar.slider(
    "End-of-line sampling rate (%)", 0, 100, 10, 5,
    help="Percentage of parts randomly inspected at the end of the line."
) / 100.0

st.sidebar.caption("💡 Deploy = making this workflow available on the factory line or hosting it online.")

# Scenario Presets + Reset
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
results = compute_metrics(
    df,
    confidence_threshold,
    manual_band,
    vibration_weight,
    acoustic_weight,
    sampling_rate
)

# --- Display KPIs ---
st.subheader("Performance Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy (%)", results["accuracy"])
col2.metric("Recall (%)", results["recall"])
col3.metric("Precision (%)", results["precision"])

col4, col5, col6 = st.columns(3)
col4.metric("False Negatives", results["false_negatives"])
col5.metric("Takt Time (sec)", results["takt_time"])
col6.metric("Jobs per Hour", results["jobs_per_hour"])

col7, col8 = st.columns(2)
col7.metric("Inspection Cost ($)", results["inspection_cost"])
col8.metric("Defect Cost ($)", results["defect_cost"])

# --- ROI inputs + KPI ---
st.subheader("ROI Estimator")
col_roi1, col_roi2, col_roi3 = st.columns(3)
baseline_defect_cost = col_roi1.number_input(
    "Baseline defect cost ($)",
    min_value=0.0, value=float(results["defect_cost"]) * 2,
)
investment_cost = col_roi2.number_input(
    "Investment cost ($)",
    min_value=0.0, value=5000.0,
)
current_defect_cost = results["defect_cost"]
roi_value = compute_roi(baseline_defect_cost, current_defect_cost, results["inspection_cost"], investment_cost)
col_roi3.metric("ROI (ratio)", roi_value if roi_value is not None else "—")

# --- Confidence distribution plot ---
st.subheader("Confidence Distribution")
fig = px.scatter(
    results["df"],
    x="part_id",
    y="confidence",
    color="label",
    title="Confidence by Part",
    labels={"confidence": "Confidence Score"}
)
st.plotly_chart(fig, use_container_width=True)

# --- Sample image with selector + label lookup ---
st.subheader("Sample Part Image")
images_dir = "qc_simulation/images"
image_files = []
if os.path.isdir(images_dir):
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

if image_files:
    selected_image = st.selectbox("Choose a part image:", sorted(image_files))
    st.image(os.path.join(images_dir, selected_image), caption=selected_image, use_column_width=True)
    part_stem = os.path.splitext(selected_image)[0]
    original_label = lookup_part_label(results["df"], part_stem)
    if original_label is not None:
        st.info(f"Original dataset label for part {part_stem}: {original_label}")
    else:
        st.warning(f"No matching record found in CSV for part {part_stem}.")
else:
    st.info("No images found in ../images. Add files like 001.png to enable image selection.")

st.divider()

# --- Participant logging ---
st.subheader("Log Your Team's Run")
team_name = st.text_input("Enter your team name:")
if st.button("Save Run"):
    if team_name.strip():
        log_run(team_name, confidence_threshold, manual_band,
                vibration_weight, acoustic_weight, sampling_rate, results)
        st.success(f"Run saved for team: {team_name}")
    else:
        st.warning("Please enter a team name before saving.")

# --- View logged runs ---
st.subheader("View Logged Runs")
logs_df = view_logs()
if logs_df.empty:
    st.info("No runs logged yet. Save a run to see it here.")
else:
    teams = logs_df["team"].unique().tolist()
    selected_team = st.selectbox("Filter by team:", ["All"] + teams)
    if selected_team != "All":
        logs_df = logs_df[logs_df["team"] == selected_team]
    sort_column = st.selectbox("Sort by column:", logs_df.columns.tolist(), index=0)
    sort_order = st.radio("Order:", ["Ascending", "Descending"], horizontal=True)
    logs_df = logs_df.sort_values(by=sort_column, ascending=(sort_order == "Ascending"))
    st.dataframe(logs_df, use_container_width=True)
    csv_data = logs_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered results as CSV", csv_data,
                       file_name="participant_runs_filtered.csv", mime="text/csv")
