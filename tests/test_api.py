import cv2
import numpy as np
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def create_dummy_fabric_image() -> bytes:
    img = np.full((256, 256, 3), 128, dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_single_inspect_endpoint():
    img_bytes = create_dummy_fabric_image()
    response = client.post(
        "/inspect?threshold=0.5",
        files={"file": ("sample_carpet.png", img_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] in ["OK", "Defective"]
    assert 0.0 <= data["anomaly_score"] <= 1.0
    assert data["inference_ms"] < 200.0

def test_batch_inspect_endpoint():
    img_bytes = create_dummy_fabric_image()
    files = [
        ("files", ("fabric_1.png", img_bytes, "image/png")),
        ("files", ("fabric_2.png", img_bytes, "image/png")),
    ]
    response = client.post("/inspect-batch?threshold=0.5", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 2
