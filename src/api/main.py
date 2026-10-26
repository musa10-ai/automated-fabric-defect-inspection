from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from src.db.models import init_db, SessionLocal, InspectionRecord
from src.model.predictor import FabricInspector

init_db()
app = FastAPI(title="Automated Fabric Defect Inspection API", version="1.0.0")
inspector = FabricInspector()

def log_to_db(res: dict):
    db = SessionLocal()
    try:
        record = InspectionRecord(**res)
        db.add(record)
        db.commit()
    finally:
        db.close()

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/inspect")
async def inspect_single(
    file: UploadFile = File(...),
    threshold: Optional[float] = Query(0.50, ge=0.0, le=1.0, description="Decision threshold")
):
    content = await file.read()
    try:
        result = inspector.inspect(content, file.filename, threshold=threshold)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    log_to_db(result)
    return result

@app.post("/inspect-batch")
async def inspect_batch(
    files: List[UploadFile] = File(...),
    threshold: Optional[float] = Query(0.50, ge=0.0, le=1.0, description="Decision threshold")
):
    results = []
    for file in files:
        content = await file.read()
        result = inspector.inspect(content, file.filename, threshold=threshold)
        log_to_db(result)
        results.append(result)
    return {"total_processed": len(results), "results": results}
