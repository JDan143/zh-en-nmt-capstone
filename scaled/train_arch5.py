from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if "SPEECHBRIDGE_RESULTS" in os.environ:
    os.environ["SPEECHBRIDGE_RESULTS"] = os.path.join(
        os.environ["SPEECHBRIDGE_RESULTS"], "arch5")
else:
    os.environ["SPEECHBRIDGE_RESULTS"] = os.path.join(
        os.environ.get("SPEECHBRIDGE_ROOT", "."), "results", "arch5")
os.makedirs(os.environ["SPEECHBRIDGE_RESULTS"], exist_ok=True)

import train

train.MODEL_MODULES["arch5_improved_transformer"] = "models.arch5_improved_transformer"

if __name__ == "__main__":
    if "--arch" not in sys.argv:
        sys.argv += ["--arch", "arch5_improved_transformer"]
    train.main()
