"""Alembic environment: deployment is an explicit migration step."""

from logging.config import fileConfig
import os
from pathlib import Path

from alembic import context
from dotenv import load_dotenv

from backend.db import make_engine
from backend.models import Base

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)
config = context.config
if config.config_file_name:
    fileConfig(config.config_file_name)
url = os.getenv("BACKEND_DATABASE_URL") or config.get_main_option("sqlalchemy.url")
target_metadata = Base.metadata


def run_migrations_offline():
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True,
                      dialect_opts={"paramstyle": "named"}, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    supplied = config.attributes.get("connection")
    if supplied is not None:
        context.configure(connection=supplied, target_metadata=target_metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()
        return
    engine = make_engine(url)
    try:
        with engine.connect() as connection:
            context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
