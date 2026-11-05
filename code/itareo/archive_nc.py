"""No-clobber archive copy (CLAUDE.md 13.4, archive rule), the Python twin of scripts/archive_nc.sh: copies each file into
the destination folder keeping its timestamps, and raises FileExistsError, copying nothing, if any destination exists."""
import shutil
from pathlib import Path


def archive_nc(dst_dir, *files):
    dst = Path(dst_dir)
    targets = [dst / Path(f).name for f in files]
    clash = [str(t) for t in targets if t.exists()]
    if clash:
        raise FileExistsError(f"archive_nc: {clash} exist, nothing copied")
    dst.mkdir(parents=True, exist_ok=True)
    for f, t in zip(files, targets):
        shutil.copy2(f, t)
    return [str(t) for t in targets]
