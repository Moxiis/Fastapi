"""Storage shim: expose helper functions for the rest of the app.

This module delegates to the SQLAlchemy-backed `app.core.db` helpers. Keeping
the original function names helps preserve existing callsites.
"""

from . import db

# AGENT: remove storage.py and just expose db functions directly here to simplify the codebase and avoid unnecessary indirection


def create_storage_tables() -> None:
    db.create_tables()


def store_raw_input(obj: dict) -> int:
    return db.store_raw_input(obj)


def store_preprocessed(obj: dict) -> int:
    return db.store_preprocessed(obj)


def store_prediction(obj: dict) -> int:
    return db.store_prediction(obj)


def add_initial_training_row(data: dict) -> int:
    return db.add_initial_training_row(data)


def get_initial_training_data(limit: int | None = None):
    return db.get_initial_training_data(limit=limit)
