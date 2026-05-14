from datetime import datetime
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="researcher")
    created_at = Column(DateTime, default=datetime.utcnow)


class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(String, primary_key=True)
    owner_id = Column(String, ForeignKey("users.id"), nullable=True)
    name = Column(String, index=True, nullable=False)
    status = Column(String, index=True, nullable=False)
    storage_path = Column(String, nullable=False)
    metadata_path = Column(String, nullable=True)
    stats = Column(JSON, default=dict)
    label_map = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    owner = relationship("User")


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    architecture = Column(String, nullable=False)
    checkpoint_path = Column(String, nullable=False)
    tokenizer_path = Column(String, nullable=True)
    metrics = Column(JSON, default=dict)
    training_dataset_id = Column(String, ForeignKey("datasets.id"), nullable=True)
    parent_model_id = Column(String, ForeignKey("model_versions.id"), nullable=True)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class TrainingJob(Base):
    __tablename__ = "training_jobs"
    id = Column(String, primary_key=True)
    dataset_id = Column(String, ForeignKey("datasets.id"), nullable=False)
    model_version_id = Column(String, ForeignKey("model_versions.id"), nullable=True)
    status = Column(String, index=True, nullable=False)
    strategy = Column(String, nullable=False)
    config = Column(JSON, default=dict)
    metrics = Column(JSON, default=dict)
    logs_uri = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(String, primary_key=True)
    dataset_id = Column(String, ForeignKey("datasets.id"), nullable=True)
    model_version_id = Column(String, ForeignKey("model_versions.id"), nullable=True)
    status = Column(String, index=True, nullable=False)
    input_uri = Column(String, nullable=False)
    raw_text = Column(Text, default="")
    corrected_text = Column(Text, default="")
    translated_text = Column(Text, default="")
    confidence = Column(Float, default=0.0)
    anomaly_score = Column(Float, default=0.0)
    unknown_symbol_ratio = Column(Float, default=0.0)
    pipeline_trace = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)


class Annotation(Base):
    __tablename__ = "annotations"
    id = Column(String, primary_key=True)
    dataset_id = Column(String, ForeignKey("datasets.id"), nullable=True)
    prediction_id = Column(String, ForeignKey("predictions.id"), nullable=True)
    image_uri = Column(String, nullable=False)
    text = Column(Text, nullable=True)
    symbols = Column(JSON, default=list)
    source = Column(String, default="human")
    quality_score = Column(Float, default=1.0)
    created_at = Column(DateTime, default=datetime.utcnow)


class EmbeddingRecord(Base):
    __tablename__ = "embeddings"
    id = Column(String, primary_key=True)
    dataset_id = Column(String, ForeignKey("datasets.id"), nullable=True)
    symbol = Column(String, nullable=True)
    script_label = Column(String, nullable=True)
    vector_index = Column(Integer, nullable=False)
    embedding_type = Column(String, nullable=False)
    metadata_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
