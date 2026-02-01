#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    try:
        import emg  # noqa: F401
        import torch  # noqa: F401
        import pandas  # noqa: F401
        import pyarrow  # noqa: F401
    except Exception as e:
        print(f"FAIL import: {e}")
        sys.exit(1)

    try:
        tmp_dir = Path("data/tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        test_path = tmp_dir / "stage0_write.txt"
        test_path.write_text("ok")
    except Exception as e:
        print(f"FAIL write: {e}")
        sys.exit(1)

    print("PASS stage0")


if __name__ == "__main__":
    main()
