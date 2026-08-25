"""P0-1 — sim 궤적 train/val/test split manifest 생성 (설계 §2·§3-3).

seed 단위 70/15/15 분할(같은 seed의 전 레이아웃을 같은 split — 교차 레이아웃 누수 차단).
결정적(고정 셔플 시드) → 같은 명령이면 같은 manifest. 산출: dataset/trajectories/split_manifest.json.

실행: uv run python train/make_traj_split.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from trajectory.split import build_manifest, parse_scene       # noqa: E402
from trajectory.sim_traj import MANIFEST_NAME                  # noqa: E402

TRAJ = ROOT / "dataset" / "trajectories"


def main():
    files = sorted(os.path.basename(f) for f in glob.glob(str(TRAJ / "*.json")))
    files = [f for f in files if f != MANIFEST_NAME]
    manifest = build_manifest(files, ratios=(0.7, 0.15, 0.15), shuffle_seed=0)

    out = TRAJ / MANIFEST_NAME
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[manifest] {out}")
    print(f"counts: {manifest['meta']['counts']} · layouts: {manifest['meta']['layouts']}")
    for split in ("train", "val", "test"):
        seeds = manifest["meta"]["seeds"][split]
        dist = Counter(parse_scene(f)[0] for f in manifest[split])
        print(f"  {split:5} seeds({len(seeds)})={seeds} · files {len(manifest[split])} · layout {dict(dist)}")
    # 누수 방어 확인 출력: 한 seed가 한 split에만 있는지.
    where = {}
    for split in ("train", "val", "test"):
        for f in manifest[split]:
            where.setdefault(parse_scene(f)[1], set()).add(split)
    leaks = {s: v for s, v in where.items() if len(v) > 1}
    print("누수 seed(2개 split 이상):", leaks or "없음")


if __name__ == "__main__":
    main()
