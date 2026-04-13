from app.core.logger import configure_logging

# initialize logging and ensure DB tables exist as early as possible
configure_logging()
from app.core.storage import create_storage_tables

create_storage_tables()

from pydantic import BaseModel
from sqlalchemy import Boolean, Column, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from fastapi import Depends, FastAPI

# 1. DATABASE SETUP
# We use SQLite because it's a simple file on your computer—no setup required!
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# 2. THE DATABASE MODEL (SQLAlchemy)
class DBTask(Base):
    __tablename__ = "tasks"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    is_completed = Column(Boolean, default=False)


# Create the actual database file
Base.metadata.create_all(bind=engine)


# 3. THE DATA SCHEMA (Pydantic) - What the user sends/receives
class TaskSchema(BaseModel):
    title: str
    is_completed: bool = False

    class Config:
        from_attributes = True


# 4. THE APP AND DEPENDENCY
app = FastAPI()


# Register ML prediction router if available
try:
    from app.api.router import router as ml_router

    app.include_router(ml_router)
except Exception:
    # If the ml package isn't present yet (or model not trained) keep original app functional
    pass


# This function opens a database connection for one request and closes it after
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 5. THE ROUTES
@app.post("/tasks/")
def create_task(task: TaskSchema, db: Session = Depends(get_db)):
    # Convert Pydantic data to SQLAlchemy Model
    new_task = DBTask(title=task.title, is_completed=task.is_completed)
    db.add(new_task)
    db.commit()
    db.refresh(new_task)
    return new_task


@app.get("/tasks/")
def read_tasks(db: Session = Depends(get_db)):
    # Ask the database for all tasks
    return db.query(DBTask).all()
