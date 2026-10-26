from pathlib import Path
import pandas as pd
import requests
import streamlit as st
from sqlalchemy import create_engine

st.set_page_config(page_title="Fabric Defect Inspection Dashboard", layout="wide")
st.title("Automated Fabric Defect Inspection Dashboard")

DB_PATH = Path("./data/inspections.db")
import os; API_URL = os.getenv("API_URL", "http://localhost:8000")

st.sidebar.header("Run Live Inspection")
threshold = st.sidebar.slider("Decision Threshold", 0.05, 0.95, 0.50, 0.05)
uploaded_files = st.sidebar.file_uploader("Upload Fabric Image(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if st.sidebar.button("Inspect Uploaded Fabric") and uploaded_files:
    files_payload = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
    res = requests.post(f"{API_URL}/inspect-batch", params={"threshold": threshold}, files=files_payload)
    if res.status_code == 200:
        st.sidebar.success(f"Processed {len(uploaded_files)} image(s)!")
    else:
        st.sidebar.error(f"API Error: {res.text}")

if DB_PATH.exists():
    engine = create_engine(f"sqlite:///{DB_PATH}")
    df = pd.read_sql("SELECT * FROM inspections ORDER BY id DESC", engine)

    if not df.empty:
        total = len(df)
        defective_df = df[df["verdict"] == "Defective"]
        defect_rate = (len(defective_df) / total) * 100
        avg_speed = df["inference_ms"].mean()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Inspected", total)
        col2.metric("Defect Rate", f"{defect_rate:.1f}%")
        col3.metric("Avg CPU Inference Speed", f"{avg_speed:.1f} ms")

        st.subheader("Recent Defective Sections (Localized Heatmaps)")
        if not defective_df.empty:
            cols = st.columns(4)
            for idx, (_, row) in enumerate(defective_df.head(8).iterrows()):
                with cols[idx % 4]:
                    if row["heatmap_path"] and Path(row["heatmap_path"]).exists():
                        st.image(row["heatmap_path"], caption=f"{row['filename']} (Score: {row['anomaly_score']:.2f})")
        else:
            st.info("No defective fabric flagged yet.")

        st.subheader("Inspection Log History")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Database is ready. Upload fabric images from datasets/MVTecAD/carpet/test in the sidebar!")
else:
    st.warning("No database found yet. Start the FastAPI server first.")
