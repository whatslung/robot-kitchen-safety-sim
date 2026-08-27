from __future__ import annotations

import json
from pathlib import Path


ALLOWED_DATASETS = frozenset({"trajectories", "trajectories_v2"})


def resolve_dataset_dir(project_root: Path, dataset: str | None) -> Path:
    name = "trajectories" if dataset is None else str(dataset)
    if name not in ALLOWED_DATASETS:
        raise ValueError(f"허용되지 않은 trajectory dataset: {name!r}")
    return Path(project_root) / "dataset" / name


def _scene_id(value: object) -> str:
    scene_id = "".join(
        character
        for character in str(value)
        if character.isalnum() or character in "-_"
    )[:80]
    if not scene_id:
        raise ValueError("scene_id가 비어 있음")
    return scene_id


def save_trajectory_scene(body: dict, project_root: Path) -> dict:
    dataset = body.get("dataset")
    root = resolve_dataset_dir(project_root, dataset)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{_scene_id(body.get('scene_id', ''))}.json"
    path.write_text(
        json.dumps(body, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return {
        "ok": True,
        "count": len(list(root.glob("*.json"))),
        "dataset": dataset or "trajectories",
        "dir": str(root),
        "file": path.name,
    }
