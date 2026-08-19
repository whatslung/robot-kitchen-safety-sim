"""sim 궤적 로더 + 스테이션 휴리스틱 테스트. 이슈 #2 3단계."""
import json

from trajectory.types import Track
from trajectory.sim_traj import load_windows, OBS, PRED
from trajectory.sim_predictors import StationHeuristicPredictor
from trajectory.evaluator import ade, fde


def _write_scene(d, seed, node_frames):
    """node_frames: {id: [(x,z,goal,gx,gz), …]} → scene JSON 하나 기록."""
    nodes = []
    for nid, frs in node_frames.items():
        frames = [{"t": round(i * 0.4, 3), "x": x, "z": z,
                   "goal": g, "gx": gx, "gz": gz, "moving": True}
                  for i, (x, z, g, gx, gz) in enumerate(frs)]
        nodes.append({"id": nid, "job": "cook", "role": "danger",
                      "discarded": False, "frames": frames})
    scene = {"scene_id": f"t_seed{seed}", "schema": 1, "seed": seed, "layout": "island",
             "wf": True, "hz": 2.5, "dt": 0.4, "steps": len(next(iter(node_frames.values()))),
             "discarded": False, "nodes": nodes}
    (d / f"{scene['scene_id']}.json").write_text(json.dumps(scene), encoding="utf-8")


def test_loader_window_shape_and_split(tmp_path):
    # 22프레임 → 22-(8+12)+1 = 3 윈도우. seed 5 는 val, seed 1 은 train.
    frs = [(0.1 * i, 0.0, "kettle", 2.0, 0.0) for i in range(22)]
    _write_scene(tmp_path, 5, {"extra_0": frs})       # val
    _write_scene(tmp_path, 1, {"extra_0": frs})       # train

    val = load_windows("val", traj_dir=tmp_path)
    train = load_windows("train", traj_dir=tmp_path)
    assert len(val) == 3 and len(train) == 3          # 분할이 겹치지 않고 각 3윈도우
    w = val[0]
    assert len(w.scene.agents[0].history) == OBS      # 관측 8
    assert len(w.gt) == PRED                           # 예측 12
    assert w.goal == (2.0, 0.0)                        # 관측 마지막 목표
    assert w.seed == 5


def test_loader_skips_discarded_nodes(tmp_path):
    frs = [(0.1 * i, 0.0, "kettle", 2.0, 0.0) for i in range(20)]
    _write_scene(tmp_path, 5, {"extra_0": frs, "extra_1": frs})
    # extra_1 을 폐기표시
    f = tmp_path / "t_seed5.json"
    d = json.loads(f.read_text(encoding="utf-8"))
    d["nodes"][1]["discarded"] = True
    f.write_text(json.dumps(d), encoding="utf-8")
    wins = load_windows("val", traj_dir=tmp_path)
    assert all(w.node_id == "extra_0" for w in wins)   # 폐기 노드 제외


def test_station_heuristic_matches_straight_line_to_goal():
    # 관측: +x 로 0.5m/s 등속. 목표는 진행방향 먼 곳(구간 내 미도달) → 등속 직진과 GT 일치.
    now, horizon, n = 0.8, 4.8, 12
    hist = [(0.4 * i, 0.2 * i, 0.0) for i in range(OBS)]        # x=0.2m/스텝 = 0.5m/s
    track = Track(0, hist)
    x0 = hist[-1][1]
    goal = (x0 + 100.0, 0.0)                            # 멀어서 horizon 안에 못 닿음(캡핑 없음)
    steps = StationHeuristicPredictor(n_steps=n).predict_steps(track, now, horizon, goal)
    gt = [(now + horizon * i / n, x0 + 0.2 * i, 0.0) for i in range(1, n + 1)]
    assert ade(steps, gt) < 0.05 and fde(steps, gt) < 0.05     # 등속 직진 = GT


def test_station_heuristic_falls_back_to_cv_when_no_goal():
    now, horizon, n = 0.8, 4.8, 12
    hist = [(0.4 * i, 0.2 * i, 0.0) for i in range(OBS)]
    steps = StationHeuristicPredictor(n_steps=n).predict_steps(Track(0, hist), now, horizon, None)
    assert len(steps) == n                             # 폴백해도 n스텝 반환
