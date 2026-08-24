from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trajectory.traj_v2 import build_manifest, validate_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="trajectory v2 dataset 감사")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT / "dataset" / "trajectories_v2",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT
        / "docs"
        / "chanwoo"
        / "results"
        / "traj-v2-manifest.json",
    )
    parser.add_argument("--code-commit")
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    code_commit = args.code_commit or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if args.manifest.exists():
        previous = json.loads(args.manifest.read_text(encoding="utf-8"))
        generated_at = previous.get("meta", {}).get("generated_at")
    else:
        generated_at = datetime.now(timezone.utc).isoformat()
    manifest = build_manifest(
        args.dataset_dir,
        code_commit=code_commit,
        generated_at=generated_at,
    )
    validate_manifest(args.dataset_dir, manifest)
    if args.write:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(manifest["meta"]["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
