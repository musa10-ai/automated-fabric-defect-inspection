# System Requirements Document: Automated Fabric Defect Inspection

## 1. Business Problem & Objective
Textile manufacturing is Pakistan's largest export sector, yet quality control on production lines remains heavily reliant on manual human visual inspection. Manual inspection is slow, inconsistent across shifts, and prone to missing subtle anomalies such as small holes, oil stains, broken threads, and color shading bands. This system provides a mid-size textile mill in Karachi with an automated visual inspection API and real-time reporting dashboard that flags defective fabric sections and logs quality metrics.

## 2. Scope
- **Approach:** Unsupervised/self-supervised anomaly detection trained strictly on defect-free fabric images using PatchCore, addressing the scarcity of labeled defect samples in production mills.
- **In-Scope (v1.0):** REST API for single and batch image inspection, localized defect heatmap generation, configurable decision thresholding, SQLite inspection logging, Streamlit manager dashboard, and Docker Compose deployment.
- **Out-of-Scope (v1.0):** Live line-scan camera hardware integration and multi-class defect categorization.

## 3. Functional Requirements
1. **Single & Batch Image Ingestion:** Expose `/inspect` and `/inspect-batch` REST API endpoints accepting fabric images.
2. **Automated Scoring & Localization:** Return a binary verdict (`OK` or `Defective`), a normalized anomaly score, and an overlaid heatmap pinpointing the defective region.
3. **Database Persistence:** Log every inspection record with a UTC timestamp, filename, anomaly score, threshold, inference latency, and heatmap file path to SQLite (PostgreSQL-ready via SQLAlchemy).
4. **Quality Reporting Dashboard:** Display total fabric images inspected, overall defect rate (%), average CPU inference latency, recent localized defect heatmaps, and full log history.

## 4. Non-Functional Requirements & Success Targets
- **Model Accuracy:** Image-level AUROC >= 0.90 and defect recall >= 90% on the MVTec AD `carpet` benchmark.
- **Latency:** Under 200 ms per image on standard CPU hardware.
- **Configurability:** Adjustable decision threshold (`0.05` to `0.95`) allowing quality managers to balance false alarms against missed defects.
- **Portability:** Full stack deployable anywhere via `docker compose up`.
