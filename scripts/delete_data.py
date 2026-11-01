"""Repository-root convenience entry point for reviewed complaint erasure."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.scripts.delete_data import main

if __name__ == "__main__":
    raise SystemExit(main())
