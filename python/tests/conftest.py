"""pytest configuration for the hochschild test suite."""
import sys
from pathlib import Path

# Ensure the parent package is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
