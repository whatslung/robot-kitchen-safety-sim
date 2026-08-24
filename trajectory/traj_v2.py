from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re


SPLIT_SEEDS = {
    "train": range(1, 31),
    "val": range(31, 41),
    "test": range(41, 51),
}
LAYOUTS = ("island_h58", "corridor", "legacy")
JOB_SETS = (
    ("cook", "carry"),
    ("cook", "lead", "wash"),
    ("prep", "wash"),
    ("cook", "carry", "wash", "lead"),
    ("prep", "cook", "carry"),
)
TELEPORT_MAX = 0.8


class DatasetAuditError(ValueError):
    pass


def split_for_seed(seed: int) -> str:
    for split, seeds in SPLIT_SEEDS.items():
        if seed in seeds:
            return split
    raise DatasetAuditError(f"범위 밖 seed: {seed}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_scene(path: Path) -> dict:
    match = re.fullmatch(
        r"(island_h58|corridor|legacy)_seed(\d+)_(\d{4})\.json", path.name
    )
    if match is None:
        raise DatasetAuditError(f"예상하지 않은 파일명: {path.name}")
    tag, raw_seed, _index = match.groups()
    seed = int(raw_seed)
    split = split_for_seed(seed)
    scene = json.loads(path.read_text(encoding="utf-8"))
    expected_layout = "island" if tag == "island_h58" else tag
    expected_half = 5.75 if tag == "island_h58" else None
    if scene.get("scene_id") != path.stem or scene.get("seed") != seed:
        raise DatasetAuditError(f"scene 식별 불일치: {path.name}")
    if scene.get("layout") != expected_layout or scene.get("half") != expected_half:
        raise DatasetAuditError(f"layout metadata 불일치: {path.name}")
    if scene.get("dt") != 0.4 or scene.get("hz") != 2.5 or scene.get("steps") != 150:
        raise DatasetAuditError(f"시간 계약 불일치: {path.name}")
    if not scene.get("room") or not scene.get("robot") or scene.get("mPerAU") is None:
        raise DatasetAuditError(f"geometry metadata 누락: {path.name}")
    nodes = scene.get("nodes", [])
    if scene.get("discarded") or any(node.get("discarded") for node in nodes):
        raise DatasetAuditError(f"폐기 scene/node 포함: {path.name}")
    jobs = tuple(node.get("job") for node in nodes)
    if jobs != JOB_SETS[(seed - 1) % len(JOB_SETS)]:
        raise DatasetAuditError(f"job 조합 불일치: {path.name}")
    max_step = 0.0
    for node in nodes:
        frames = node.get("frames", [])
        if len(frames) != 150 or any(
            abs(frame["t"] - index * 0.4) > 1e-6
            for index, frame in enumerate(frames)
        ):
            raise DatasetAuditError(
                f"frame 계약 불일치: {path.name}/{node.get('id')}"
            )
        for previous, current in zip(frames, frames[1:]):
            step = math.hypot(
                current["x"] - previous["x"], current["z"] - previous["z"]
            )
            max_step = max(max_step, step)
            if step > TELEPORT_MAX:
                raise DatasetAuditError(
                    f"프레임 이동량 초과: {path.name}/{node.get('id')} {step:.4f}m"
                )
    return {
        "name": path.name,
        "seed": seed,
        "layout": tag,
        "split": split,
        "jobs": list(jobs),
        "people": len(nodes),
        "frames": 150,
        "dt": 0.4,
        "max_step": round(max_step, 4),
        "has_room": True,
        "has_robot": True,
        "mPerAU": scene["mPerAU"],
        "sha256": sha256_file(path),
    }


def summarize_dataset(dataset_dir: Path, manifest: dict) -> dict:
    summary = {}
    for split in ("train", "val"):
        people = 0
        eligible = 0
        positive = 0
        job_sets = Counter()
        for name in manifest[split]:
            scene = json.loads((dataset_dir / name).read_text(encoding="utf-8"))
            people += len(scene["nodes"])
            job_sets[tuple(node["job"] for node in scene["nodes"])] += 1
            robot_x, robot_z = scene["robot"]["x"], scene["robot"]["z"]
            for node in scene["nodes"]:
                frames = node["frames"]
                for start in range(len(frames) - (8 + 12) + 1):
                    current = frames[start + 7]
                    if math.hypot(
                        current["x"] - robot_x, current["z"] - robot_z
                    ) < 3.10:
                        continue
                    eligible += 1
                    future = frames[start + 8 : start + 12]
                    positive += any(
                        math.hypot(frame["x"] - robot_x, frame["z"] - robot_z)
                        < 3.10
                        for frame in future
                    )
        summary[split] = {
            "scenes": len(manifest[split]),
            "people": people,
            "job_sets": {
                "|".join(key): count for key, count in sorted(job_sets.items())
            },
            "safety_eligible": eligible,
            "safety_positive": positive,
        }
    summary["test"] = {"scenes": len(manifest["test"]), "labels_locked": True}
    return summary


def build_manifest(
    dataset_dir: Path, code_commit: str, generated_at: str | None = None
) -> dict:
    dataset_dir = Path(dataset_dir)
    expected = {
        (seed, layout) for seed in range(1, 51) for layout in LAYOUTS
    }
    rows = [inspect_scene(path) for path in sorted(dataset_dir.glob("*.json"))]
    actual = {(row["seed"], row["layout"]) for row in rows}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra or len(rows) != 150:
        raise DatasetAuditError(
            f"전수 조합 누락={missing} 추가={extra} files={len(rows)}"
        )
    names = {
        split: sorted(row["name"] for row in rows if row["split"] == split)
        for split in SPLIT_SEEDS
    }
    manifest = {
        "schema": 2,
        "meta": {
            "dataset": "traj-v2",
            "code_commit": code_commit,
            "generated_at": generated_at,
            "layouts": list(LAYOUTS),
            "generation": {
                "urls": {
                    "island_h58": "sim.html?layout=island",
                    "corridor": "sim.html?layout=corridor",
                    "legacy": "sim.html?layout=legacy",
                },
                "command": (
                    "__sim.trajRun({scenes:50,seed:1,seconds:60,"
                    "dataset:'trajectories_v2'})"
                ),
            },
            "seeds": {
                split: list(seeds) for split, seeds in SPLIT_SEEDS.items()
            },
            "counts": {split: len(names[split]) for split in names},
        },
        **names,
        "files": {row["name"]: row for row in rows},
    }
    manifest["summary"] = summarize_dataset(dataset_dir, manifest)
    return manifest


def validate_manifest(dataset_dir: Path, manifest: dict) -> None:
    dataset_dir = Path(dataset_dir)
    current = {path.name for path in dataset_dir.glob("*.json")}
    recorded = set(manifest["files"])
    if current != recorded:
        raise DatasetAuditError(
            "파일 집합 불일치 "
            f"누락={sorted(recorded - current)} 추가={sorted(current - recorded)}"
        )
    changed = [
        name
        for name, row in manifest["files"].items()
        if sha256_file(dataset_dir / name) != row["sha256"]
    ]
    if changed:
        raise DatasetAuditError("SHA-256 불일치: " + ", ".join(changed))
    for split, seeds in SPLIT_SEEDS.items():
        expected_names = sorted(
            name
            for name, row in manifest["files"].items()
            if row["seed"] in seeds and row["split"] == split
        )
        if manifest.get(split) != expected_names or len(expected_names) != len(
            seeds
        ) * len(LAYOUTS):
            raise DatasetAuditError(f"split membership 불일치: {split}")
    if manifest.get("meta", {}).get("counts") != {
        "train": 90,
        "val": 30,
        "test": 30,
    }:
        raise DatasetAuditError("split count 불일치")
