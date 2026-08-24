from pathlib import Path


SIM = (Path(__file__).parents[1] / "sim.html").read_text(encoding="utf-8")


def test_corridor_has_distinct_trajectory_layout_and_tag():
    assert 'CORRIDOR ? "corridor"' in SIM
    assert 'lay === "island" ? "island_h"' in SIM


def test_traj_run_accepts_only_v1_or_v2_dataset():
    assert 'new Set(["trajectories","trajectories_v2"])' in SIM
    assert "dataset: datasetName" in SIM


def test_collection_log_names_target_dataset():
    assert '"· dataset", datasetName' in SIM


def test_collection_button_passes_selected_dataset():
    assert 'id="trajDataset"' in SIM
    assert 'dataset: $("trajDataset").value' in SIM
