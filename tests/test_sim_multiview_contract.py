import re
from pathlib import Path


SIM = Path(__file__).resolve().parents[1] / "sim.html"


def _source():
    return SIM.read_text(encoding="utf-8")


def test_approved_four_plus_one_camera_poses_are_factory_defaults():
    source = _source()
    expected = {
        "mvNW": ("-2.75,2.65,-2.75", "-3.15,0,-1.14", "1.326450"),
        "mvNE": ("2.75,2.65,-2.75", "4.37,0,-2.43", "1.326450"),
        "mvSW": ("-2.75,2.65,2.75", "-2.23,0,4.32", "1.326450"),
        "mvSE": ("2.75,2.65,2.75", "2.31,0,1.15", "1.326450"),
        "mvCenter": ("-1.10,2.65,3.80", "-1.10,0.80,0.80", "1.221730"),
    }
    for camera_id, (position, target, fov) in expected.items():
        pattern = (
            rf'id:"{camera_id}"[^\n]*pos:new V3\({re.escape(position)}\), '
            rf'tgt:new V3\({re.escape(target)}\), fov:{fov}'
        )
        assert re.search(pattern, source), camera_id
    assert source.index('id:"mvNW"') < source.index("const CAM_FACTORY")


def test_multiview_scheduler_is_weighted_single_flight_with_stale_rejection():
    source = _source()
    assert "const MV_SCHED" in source
    assert 'id:"mvNW", hz:4' in source
    assert 'id:"mvNE", hz:4' in source
    assert 'id:"mvSW", hz:4' in source
    assert 'id:"mvSE", hz:4' in source
    assert 'id:"mvCenter", hz:6' in source
    assert "if (MV_SCHED.busy) return" in source
    assert "seq:++MV_SCHED.seq" in source
    assert "payload.seq <= MV_SCHED.lastAppliedSeq" in source
    assert "receivedAt - capturedAt > MV_MAX_RESPONSE_AGE_MS" in source
    assert "track.age_ms || 0) + transitAge" in source
    assert "MV_SCHED.dropped++" in source


def test_calibration_bootstrap_uses_fixed_floor_anchors():
    source = _source()
    assert "async function mvBootstrapCalibration" in source
    assert 'mvEndpoint("/calibrate")' in source
    assert "MV_FLOOR_ANCHORS" in source
    assert "valid_world_polygon" in source
    anchor_match = re.search(r"const MV_FLOOR_ANCHORS = \[(.*?)\];", source, re.S)
    assert anchor_match
    assert anchor_match.group(1).count("new V3(") >= 6
    bootstrap = re.search(
        r"async function mvBootstrapCalibration\([^)]*\)\s*\{(.*?)\n\}", source, re.S
    )
    assert bootstrap
    assert "await mvPostTrackReset()" in bootstrap.group(1)
    assert 'mvEndpoint("/tracks/reset")' in source
    assert "calibrationSignature" in bootstrap.group(1)


def test_global_response_handler_never_reads_sim_person_ground_truth():
    source = _source()
    match = re.search(
        r"function mvHandleGlobalResponse\([^)]*\)\s*\{(.*?)\n\}", source, re.S
    )
    assert match
    handler = match.group(1)
    assert "global_tracks" in handler
    assert "MIL.tracks = nextTracks" in handler
    for forbidden in ("person.node.position", "EXTRAS", "milAssocTracks"):
        assert forbidden not in handler


def test_multiview_prediction_is_display_only():
    source = _source()
    match = re.search(r"async function mvMaybePredict\([^)]*\)\s*\{(.*?)\n\}", source, re.S)
    assert match
    body = match.group(1)
    assert 'mvEndpoint("/predict")' in body
    assert "MV_SCHED.predictions" in body
    assert "x:track.x" in body and "vx:track.vx" in body
    assert "stop_radius:SCALE.au(SAFE.stopR)" in body
    assert "SCALE.m(row.x)" not in body
    for forbidden in ("avoidDecide", "SAFE.factor", "state.seqT", "robot.position"):
        assert forbidden not in body


def test_multiview_capture_skips_dataset_shader_wait_path():
    source = _source()
    assert re.search(r'new BABYLON\.RenderTargetTexture\(\s*"mvSensorRT"', source)
    match = re.search(
        r"async function mvCaptureDataURL\([^)]*\)\s*\{(.*?)\n\}\n\nfunction mvHandleGlobalResponse",
        source,
        re.S,
    )
    assert match
    assert "renderWith(" not in match.group(1)
    assert "mvRenderCameraFast(" in match.group(1)


def test_camera_changes_and_scene_reset_invalidate_multiview_state():
    source = _source()
    assert 'const CAM_STORE = "hkr.sim.cams.v7"' in source
    assert "MV_CAMERA_IDS" in source
    assert "mvCalibrationSignature" in source
    scheduler = re.search(r"function mvSchedulerTick\([^)]*\)\s*\{(.*?)\n\}", source, re.S)
    assert scheduler
    assert "calibrationSignature" in scheduler.group(1)
    reset_handler = re.search(
        r'\$\("resetBtn"\)\.addEventListener\("click", \(\) => \{(.*?)\n\}\);',
        source,
        re.S,
    )
    assert reset_handler
    assert "mvRequestTrackReset()" in reset_handler.group(1)


def test_environment_randomization_registers_late_loaded_props():
    source = _source()
    match = re.search(
        r"function randomizeEnvironment\(\)\s*\{(.*?)\n\}", source, re.S
    )
    assert match
    body = match.group(1)
    assert "ENV_RAND.base = ENV_RAND.base || {}" in body
    assert "if (!ENV_RAND.base[h.name])" in body
