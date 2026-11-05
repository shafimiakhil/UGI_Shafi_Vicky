"""CLAUDE.md 13.5: every script refuses to write to an existing output path."""
from pathlib import Path


def refuse_existing(*paths):
    """Raises FileExistsError, naming them, if any of the paths exists."""
    found = [str(p) for p in paths if Path(p).exists()]
    if found:
        raise FileExistsError(f"refusing to overwrite: {found}")
