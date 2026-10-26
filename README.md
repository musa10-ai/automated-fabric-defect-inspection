# Automated Fabric Defect Inspection System

[![CI/CD Pipeline](https://github.com/musa10-ai/automated-fabric-defect-inspection/actions/workflows/ci.yml/badge.svg)](https://github.com/musa10-ai/automated-fabric-defect-inspection/actions)

An end-to-end computer vision pipeline for real-time anomaly detection in industrial manufacturing. This system identifies fabric defects using a PatchCore model and provides a REST API and Streamlit dashboard for quality control monitoring.

## Engineering Highlights
* **High-Performance CPU Inference**: Engineered a memory bank subsampling technique to guarantee **<200ms** inference latency per image on standard CPUs without requiring GPU acceleration.
* **Resilient CI/CD Pipeline**: Configured automated GitHub Actions for headless `pytest` validation. Implemented a defensive CI mock fallback to ensure the pipeline passes without hosting massive binary model weights in the repository.
* **Containerized Deployment**: Built lightweight Docker environments utilizing `opencv-python-headless` and CPU-optimized PyTorch wheels to minimize image bloat and eliminate cross-platform dependency conflicts.

## Tech Stack
* **Backend:** FastAPI, Python 3.11
* **Frontend:** Streamlit
* **Computer Vision:** PyTorch, Anomalib (PatchCore), OpenCV
* **DevOps:** Docker, GitHub Actions, Pytest

## Local Deployment (Docker)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/musa10-ai/automated-fabric-defect-inspection.git
