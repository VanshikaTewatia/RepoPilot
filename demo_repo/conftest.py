"""Demo repository pytest setup."""

import sys
from pathlib import Path

# Add src to sys.path
src_dir = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(src_dir))
