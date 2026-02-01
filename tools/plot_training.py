#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="data/tmp/checkpoints/train_log.json")
    ap.add_argument("--out", default="data/tmp/plots/loss.png")
    ap.add_argument("--window", type=int, default=10, help="moving average window")
    ap.add_argument("--show", action="store_true", help="display the figure")
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception:
        raise SystemExit("matplotlib 未安装，请先 pip install matplotlib")

    log_path = Path(args.log)
    if not log_path.exists():
        raise SystemExit(f"log not found: {log_path}")

    data = json.loads(log_path.read_text())
    losses = data.get("losses", [])
    if not losses:
        raise SystemExit("log 中没有 losses")

    # moving average
    if args.window > 1:
        ma = []
        for i in range(len(losses)):
            start = max(0, i - args.window + 1)
            ma.append(sum(losses[start : i + 1]) / float(i - start + 1))
    else:
        ma = losses

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(losses, label="loss", alpha=0.4)
    plt.plot(ma, label=f"ma({args.window})")
    plt.title("Training Loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)

    if args.show:
        plt.show()

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
