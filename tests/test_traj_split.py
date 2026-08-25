"""P0-1 — seed 단위 train/val/test 분할 순수 로직 (설계 §2·§6).

같은 seed의 모든 레이아웃 변형이 한 split에만 있어야 교차 레이아웃 누수가 없다.
"""
from __future__ import annotations

import pytest

from trajectory.split import parse_scene, assign_seeds, build_manifest


def test_parse_scene():
    assert parse_scene("island_seed1_0000.json") == ("island", 1)
    assert parse_scene("island_h58_seed12_0011.json") == ("island_h58", 12)
    assert parse_scene("legacy_seed40_0119.json") == ("legacy", 40)


def test_parse_scene_non_sim_returns_none():
    assert parse_scene("real_test_sample.json") is None


def test_assign_seeds_counts_70_15_15():
    a = assign_seeds(list(range(1, 41)), ratios=(0.7, 0.15, 0.15), shuffle_seed=0)
    assert (len(a["train"]), len(a["val"]), len(a["test"])) == (28, 6, 6)


def test_assign_seeds_partition_no_overlap_full_cover():
    a = assign_seeds(list(range(1, 41)), shuffle_seed=0)
    tr, va, te = set(a["train"]), set(a["val"]), set(a["test"])
    assert tr & va == set() and tr & te == set() and va & te == set()
    assert tr | va | te == set(range(1, 41))


def test_assign_seeds_deterministic():
    a = assign_seeds(list(range(1, 41)), shuffle_seed=0)
    b = assign_seeds(list(range(1, 41)), shuffle_seed=0)
    assert a == b


def test_assign_seeds_shuffle_seed_changes_partition():
    a = assign_seeds(list(range(1, 41)), shuffle_seed=0)
    b = assign_seeds(list(range(1, 41)), shuffle_seed=1)
    assert a != b                    # 다른 셔플 시드 → 다른 배정(거의 확실)


def test_assign_seeds_input_order_independent():
    a = assign_seeds([3, 1, 2, 5, 4, 6, 7, 8, 9, 10], shuffle_seed=0)
    b = assign_seeds([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], shuffle_seed=0)
    assert a == b                    # 정렬 후 셔플이라 입력 순서 무관


def _files():
    files = []
    for seed in range(1, 41):
        for lay in ("island", "island_h58", "legacy"):
            files.append(f"{lay}_seed{seed}_{seed:04d}.json")
    files.append("real_test_sample.json")     # sim 아님 → 제외되어야
    return files


def test_build_manifest_counts_and_excludes_non_sim():
    m = build_manifest(_files(), ratios=(0.7, 0.15, 0.15), shuffle_seed=0)
    assert m["meta"]["counts"] == {"train": 84, "val": 18, "test": 18}
    all_files = m["train"] + m["val"] + m["test"]
    assert "real_test_sample.json" not in all_files
    assert len(all_files) == 120


def test_build_manifest_no_seed_leak_across_splits():
    m = build_manifest(_files(), shuffle_seed=0)
    where = {}
    for split in ("train", "val", "test"):
        for f in m[split]:
            _lay, seed = parse_scene(f)
            where.setdefault(seed, set()).add(split)
    # 한 seed의 모든 파일이 정확히 한 split에만.
    assert all(len(s) == 1 for s in where.values())


def test_build_manifest_layout_stratified():
    m = build_manifest(_files(), shuffle_seed=0)
    for split in ("train", "val", "test"):
        layouts = {parse_scene(f)[0] for f in m[split]}
        assert layouts == {"island", "island_h58", "legacy"}


def test_build_manifest_deterministic_and_sorted():
    m1 = build_manifest(_files(), shuffle_seed=0)
    m2 = build_manifest(_files(), shuffle_seed=0)
    assert m1 == m2
    for split in ("train", "val", "test"):
        assert m1[split] == sorted(m1[split])       # 재현 diff 없게 정렬 저장
