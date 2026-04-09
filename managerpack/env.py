"""Environment loading for dev/prod credential management."""

import os
from pathlib import Path

import click


def load_env(env: str = "dev") -> None:
    """Load .env.dev or .env.prod into os.environ.

    Values already set in the environment are NOT overwritten,
    so explicit exports always win.
    """
    env_file = Path(f".env.{env}")
    if not env_file.exists():
        # Walk up to repo root
        env_file = Path(__file__).resolve().parent.parent / f".env.{env}"
    if not env_file.exists():
        raise click.ClickException(f"Missing {env_file}")
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        if key and value and key not in os.environ:
            os.environ[key] = value
