"""Put src/ on the path so `pytest tests/` works from a clean checkout."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))
