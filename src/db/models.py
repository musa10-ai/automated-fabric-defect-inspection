from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy import Column, Integer, String, Float, DateTime, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DB_DIR = Path("./data")
DB_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_URL = f"sqlite:///{DB_DIR / 'inspections.db'}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class InspectionRecord(Base):
    __tablename__ = "inspections"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    filename = Column(String, nullable=False)
    verdict = Column(String, nullable=False)
    anomaly_score = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    inference_ms = Column(Float, nullable=False)
    heatmap_path = Column(String, nullable=True)

def init_db():
    Base.metadata.create_all(bind=engine)
