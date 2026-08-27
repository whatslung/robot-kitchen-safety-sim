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


def test_collection_button_passes_start_seed():
    assert 'id="trajSeed"' in SIM
    assert 'seed: +$("trajSeed").value' in SIM


def test_job_set_selection_depends_on_seed_not_batch_position():
    assert 'TRAJ.jobSets[((base + k) - 1) % TRAJ.jobSets.length]' in SIM


def test_job_spawn_enforces_role_radius_before_path_creation():
    expected = (
        "enforceExtraRoleRadius(P);\n"
        "  extraNextTarget(P);"
    )
    assert expected in SIM


def test_role_radius_and_people_separation_stay_inside_room():
    assert "function clampExtraToRoom(P)" in SIM
    assert "n.x = clamp(n.x, room.x0, room.x1);" in SIM
    assert "n.z = clamp(n.z, room.z0, room.z1);" in SIM
    assert "for (const P of EXTRAS) clampExtraToRoom(P);" in SIM
