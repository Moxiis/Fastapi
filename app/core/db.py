from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Generator, Optional, Tuple

import numpy as np
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
    select,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from .config import settings

# SQLAlchemy setup
connect_args = {}
if settings.database_url.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(settings.database_url, connect_args=connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


# AGENT: removed rawinput
class RawInput(Base):
    __tablename__ = "raw_inputs"
    id = Column(Integer, primary_key=True, index=True)
    payload = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# AGENT: remove Preprocessed
class Preprocessed(Base):
    __tablename__ = "preprocessed"
    id = Column(Integer, primary_key=True, index=True)
    features = Column(JSON, nullable=False)
    model_version = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# AGENT: change input from Json to tabular data with columns for each feature and target to support more structured storage and querying
class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    input = Column(JSON, nullable=True)
    features = Column(JSON, nullable=True)
    prediction = Column(Float, nullable=True)
    model_version = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class InitialTrainingData(Base):
    __tablename__ = "initial_training"
    id = Column(Integer, primary_key=True, index=True)
    medinc = Column(Float, nullable=False)
    houseage = Column(Float, nullable=False)
    averooms = Column(Float, nullable=False)
    avebedrms = Column(Float, nullable=False)
    population = Column(Float, nullable=False)
    aveoccup = Column(Float, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    target = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


def create_tables() -> None:
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def store_raw_input(obj: dict) -> int:
    try:
        with get_session() as db:
            row = RawInput(payload=obj)
            db.add(row)
            db.flush()
            return int(row.id)
    except SQLAlchemyError:
        raise


def store_preprocessed(obj: dict) -> int:
    try:
        with get_session() as db:
            row = Preprocessed(
                features=obj.get("features"), model_version=obj.get("model_version")
            )
            db.add(row)
            db.flush()
            return int(row.id)
    except SQLAlchemyError:
        raise


def store_prediction(obj: dict) -> int:
    try:
        with get_session() as db:
            row = Prediction(
                input=obj.get("input"),
                features=obj.get("features"),
                prediction=obj.get("prediction"),
                model_version=obj.get("model_version_id") or obj.get("model_version"),
            )
            db.add(row)
            db.flush()
            return int(row.id)
    except SQLAlchemyError:
        raise


def add_initial_training_row(data: dict) -> int:
    try:
        with get_session() as db:
            row = InitialTrainingData(
                medinc=data["medinc"],
                houseage=data["houseage"],
                averooms=data["averooms"],
                avebedrms=data["avebedrms"],
                population=data["population"],
                aveoccup=data["aveoccup"],
                latitude=data["latitude"],
                longitude=data["longitude"],
                target=data.get("target"),
            )
            db.add(row)
            db.flush()
            return int(row.id)
    except SQLAlchemyError:
        raise


def get_initial_training_data(
    limit: Optional[int] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    with get_session() as db:
        stmt = select(InitialTrainingData)
        if limit:
            stmt = stmt.limit(limit)
        rows = db.execute(stmt).scalars().all()
        if not rows:
            return None

        X = []
        y = []
        for r in rows:
            X.append(
                [
                    r.medinc,
                    r.houseage,
                    r.averooms,
                    r.avebedrms,
                    r.population,
                    r.aveoccup,
                    r.latitude,
                    r.longitude,
                ]
            )
            y.append(r.target if r.target is not None else np.nan)

        return np.asarray(X, dtype=float), np.asarray(y, dtype=float)
