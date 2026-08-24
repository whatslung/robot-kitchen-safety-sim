import json

import pytest

from trajectory.collection import resolve_dataset_dir, save_trajectory_scene


def _scene(dataset=None):
    out = {
        "scene_id": "island_h58_seed1_0000",
        "schema": 1,
        "seed": 1,
        "nodes": [],
    }
    if dataset is not None:
        out["dataset"] = dataset
    return out


def test_default_dataset_stays_v1(tmp_path):
    result = save_trajectory_scene(_scene(), tmp_path)

    assert result["dataset"] == "trajectories"
    assert (
        tmp_path
        / "dataset"
        / "trajectories"
        / "island_h58_seed1_0000.json"
    ).exists()


def test_v2_dataset_uses_separate_directory(tmp_path):
    result = save_trajectory_scene(_scene("trajectories_v2"), tmp_path)
    path = (
        tmp_path
        / "dataset"
        / "trajectories_v2"
        / "island_h58_seed1_0000.json"
    )

    assert result["dataset"] == "trajectories_v2"
    assert json.loads(path.read_text(encoding="utf-8"))["seed"] == 1


@pytest.mark.parametrize("name", ["../trajectories", "x/y", "trajectories_v3", ""])
def test_unknown_or_path_dataset_is_rejected(tmp_path, name):
    with pytest.raises(ValueError, match="허용되지 않은 trajectory dataset"):
        resolve_dataset_dir(tmp_path, name)
