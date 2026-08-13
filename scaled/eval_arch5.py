from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import evaluate

evaluate.MODEL_MODULES["arch5_improved_transformer"] = "models.arch5_improved_transformer"

if __name__ == "__main__":
    evaluate.main()
