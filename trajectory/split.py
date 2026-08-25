"""P0-1 — seed 단위 train/val/test 분할 (설계 §2, docs/chanwoo/specs/2026-08-24-traj-split-ci-design.md).

분할 단위는 **seed**다. 한 seed의 모든 레이아웃 변형(island·island_h58·legacy)은 같은 split에
들어간다 — 같은 seed는 레이아웃만 다른 근사 중복이라, 파일 단위로 나누면 train/test에 near-duplicate
가 새어 누수가 되기 때문. seed마다 세 레이아웃을 다 가지므로 레이아웃 계층화도 자동으로 성립한다.
"""
from __future__ import annotations

import random
import re
from collections import defaultdict

_SCENE_RE = re.compile(r"^(?P<layout>.+?)_seed(?P<seed>\d+)_\d+\.json$")


def parse_scene(name):
    """'island_h58_seed12_0011.json' → ('island_h58', 12). sim scene 아니면 None."""
    m = _SCENE_RE.match(name)
    return (m.group("layout"), int(m.group("seed"))) if m else None


def assign_seeds(seeds, ratios=(0.7, 0.15, 0.15), shuffle_seed=0):
    """seed 목록 → {'train','val','test': [seed…]}. 정렬 후 고정 시드로 셔플(입력 순서 무관·결정적).

    개수 = round(n·r_train), round(n·r_val), 나머지 = test. 무중복·전수 커버.
    """
    uniq = sorted(set(seeds))
    n = len(uniq)
    order = uniq[:]
    random.Random(shuffle_seed).shuffle(order)
    n_train = round(n * ratios[0])
    n_val = round(n * ratios[1])
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    return {
        "train": sorted(order[:n_train]),
        "val": sorted(order[n_train:n_train + n_val]),
        "test": sorted(order[n_train + n_val:]),
    }


def build_manifest(files, ratios=(0.7, 0.15, 0.15), shuffle_seed=0):
    """파일 목록 → split manifest dict. sim scene 아닌 파일(예: real_test_sample)은 제외.

    반환: {"meta": {...}, "train": [file…], "val": [...], "test": [...]}  (각 split 파일명 정렬).
    """
    by_seed = defaultdict(list)          # seed → [file…]
    layouts = set()
    for f in files:
        parsed = parse_scene(f)
        if parsed is None:
            continue
        layout, seed = parsed
        by_seed[seed].append(f)
        layouts.add(layout)

    seed_split = assign_seeds(list(by_seed.keys()), ratios, shuffle_seed)
    out = {"train": [], "val": [], "test": []}
    for split, seeds in seed_split.items():
        for s in seeds:
            out[split].extend(by_seed[s])
    for split in out:
        out[split].sort()

    out["meta"] = {
        "unit": "seed",
        "ratios": list(ratios),
        "shuffle_seed": shuffle_seed,
        "layouts": sorted(layouts),
        "seeds": seed_split,
        "counts": {k: len(out[k]) for k in ("train", "val", "test")},
    }
    return out
