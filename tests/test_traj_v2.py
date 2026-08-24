import json
from pathlib import Path
import subprocess
import sys

import pytest

from trajectory.traj_v2 import build_manifest, validate_manifest, DatasetAuditError


LAYOUT_META = {
    "island_h58": {"layout": "island", "half": 5.75},
    "corridor": {"layout": "corridor", "half": None},
    "legacy": {"layout": "legacy", "half": None},
}
JOB_SETS = [
    ("cook", "carry"),
    ("cook", "lead", "wash"),
    ("prep", "wash"),
    ("cook", "carry", "wash", "lead"),
    ("prep", "cook", "carry"),
]


def _write_v2(root, seed, tag):
    jobs = JOB_SETS[(seed - 1) % 5]
    meta = LAYOUT_META[tag]
    frames = [
        {
            "t": round(i * 0.4, 3),
            "x": 0.01 * i,
            "z": 0.0,
            "goal": None,
            "gx": None,
            "gz": None,
            "moving": True,
        }
        for i in range(150)
    ]
    scene = {
        "scene_id": f"{tag}_seed{seed}_{seed - 1:04d}",
        "schema": 1,
        "seed": seed,
        "layout": meta["layout"],
        "half": meta["half"],
        "room": {"x0": -5, "x1": 5, "z0": -5, "z1": 5},
        "robot": {"x": -1.1, "z": 0.815},
        "mPerAU": 1.0,
        "wf": True,
        "hz": 2.5,
        "dt": 0.4,
        "steps": 150,
        "discarded": False,
        "nodes": [
            {
                "id": f"extra_{i}",
                "job": job,
                "role": "danger",
                "discarded": False,
                "frames": frames,
            }
            for i, job in enumerate(jobs)
        ],
    }
    (root / f"{scene['scene_id']}.json").write_text(
        json.dumps(scene), encoding="utf-8"
    )


def _full_dataset(root):
    root.mkdir()
    for seed in range(1, 51):
        for tag in LAYOUT_META:
            _write_v2(root, seed, tag)


def test_manifest_has_fixed_90_30_30_split_and_150_hashes(tmp_path):
    data = tmp_path / "v2"
    _full_dataset(data)

    manifest = build_manifest(
        data, code_commit="abc123", generated_at="2026-08-24T00:00:00Z"
    )

    assert manifest["meta"]["counts"] == {"train": 90, "val": 30, "test": 30}
    assert len(manifest["files"]) == 150
    assert set(manifest["meta"]["layouts"]) == {
        "island_h58",
        "corridor",
        "legacy",
    }
    assert set(manifest["summary"]["train"]) >= {
        "scenes",
        "people",
        "job_sets",
        "safety_eligible",
        "safety_positive",
    }
    assert manifest["summary"]["test"] == {"scenes": 30, "labels_locked": True}
    validate_manifest(data, manifest)


def test_manifest_detects_changed_file(tmp_path):
    data = tmp_path / "v2"
    _full_dataset(data)
    manifest = build_manifest(data, code_commit="abc123")
    path = data / manifest["train"][0]
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(DatasetAuditError, match="SHA-256"):
        validate_manifest(data, manifest)


def test_manifest_rejects_missing_layout(tmp_path):
    data = tmp_path / "v2"
    _full_dataset(data)
    next(data.glob("corridor_seed1_*.json")).unlink()

    with pytest.raises(DatasetAuditError, match="누락"):
        build_manifest(data, code_commit="abc123")


def test_audit_cli_resolves_project_packages():
    root = Path(__file__).parents[1]

    result = subprocess.run(
        [sys.executable, str(root / "train" / "audit_traj_v2.py"), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
