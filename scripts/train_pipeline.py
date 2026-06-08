from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd):
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    run([sys.executable, "scripts/verify_dataset.py"])
    run([sys.executable, "-m", "src.train"])
    run([sys.executable, "-m", "src.evaluate"])
    print("\nPipeline complete. Run Streamlit with: streamlit run app.py")


if __name__ == "__main__":
    main()
