#!/usr/bin/env python3
import argparse
from pathlib import Path
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps_out", type=int, default=60)
    ap.add_argument("--clip_sec", type=float, default=4.0)
    ap.add_argument("--stride_sec", type=float, default=2.0)
    ap.add_argument("--lowpass_hz", type=float, default=6.0)
    ap.add_argument("--use_delta_q", action="store_true", default=True)
    ap.add_argument("--joints_expected", type=int, default=4)
    ap.add_argument("--angle_unit", default="rad")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = []
    for idx, csv_path in enumerate(sorted(raw_dir.glob("*.csv"))):
        demo_id = csv_path.stem
        # try to infer task_type from filename suffix: demo_xxxx_task
        parts = demo_id.split("_")
        task_type = parts[-1] if len(parts) >= 3 else "idle"
        valence = 0.2 + 0.6 * (idx % 2)
        arousal = 0.2 + 0.6 * ((idx + 1) % 2)
        items.append(
            {
                "demo_id": demo_id,
                "csv_path": str(csv_path.as_posix()),
                "emotion": {"valence": float(valence), "arousal": float(arousal), "text": ""},
                "context": {
                    "task_type": task_type,
                    "human_distance": 1.2,
                    "target_xyz": [0.6, 0.0, 1.0],
                    "speed_scale": 1.0,
                    "duration_sec": float(args.clip_sec),
                },
            }
        )

    manifest = {
        "fps_out": args.fps_out,
        "clip_sec": args.clip_sec,
        "stride_sec": args.stride_sec,
        "lowpass_hz": args.lowpass_hz,
        "use_delta_q": args.use_delta_q,
        "joints_expected": args.joints_expected,
        "angle_unit": args.angle_unit,
        # Values listed here are projected into a fixed numeric vector for model conditioning.
        "context_num_keys": [
            "human_distance",
            "target_xyz",
            "speed_scale",
            "duration_sec",
        ],
        "items": items,
    }

    with out_path.open("w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False, allow_unicode=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
