from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
import time

SQLALCHEMY_DATABASE_URL = "sqlite:///./system.db"

# Check same thread = False is needed for SQLite with FastAPI
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Device(Base):
    __tablename__ = "devices"

    id = Column(String, primary_key=True, index=True)
    type = Column(String)
    status = Column(String)
    last_seen = Column(Float)
    config = Column(JSON, nullable=True)

class SystemEvent(Base):
    __tablename__ = "system_events"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(Float, default=time.time)
    event_type = Column(String)
    details = Column(String)

class MetricSnapshot(Base):
    __tablename__ = "metric_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(Float, default=time.time)
    tflops_used = Column(Float)
    memory_used_gb = Column(Float)

class SafetyZone(Base):
    __tablename__ = "safety_zones"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True)
    zone_type = Column(String)  # "KEEP_OUT", "SLOW_DOWN"
    coordinates_json = Column(String)  # JSON list of [x, y] points
    is_active = Column(Boolean, default=True)

class SafetyConfig(Base):
    __tablename__ = "safety_config"
    id = Column(Integer, primary_key=True, index=True)
    human_sensitivity = Column(Float, default=0.8)
    stop_distance_m = Column(Float, default=1.5)
    max_speed_limit = Column(Float, default=2.0)

def init_db():
    Base.metadata.create_all(bind=engine)

    # Seed default config if empty
    db = SessionLocal()
    if not db.query(SafetyConfig).first():
        db.add(SafetyConfig(id=1))
        db.commit()
    db.close()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
