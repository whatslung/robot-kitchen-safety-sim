from pathlib import Path


SIM = Path(__file__).parents[1] / "sim.html"


def sim_source() -> str:
    return SIM.read_text(encoding="utf-8")


def test_sim_loads_safety_motion_before_inline_controller():
    text = sim_source()
    assert '<script src="./safety_motion.js"></script>' in text
    assert text.index("safety_motion.js") < text.index("const SAFE =")


def test_default_avoidance_uses_simulator_planned_paths():
    text = sim_source()
    assert "function plannedPeopleOccupancy" in text
    assert "SafetyMotion.samplePlannedPath" in text
    assert "person.path" in text
    assert "P.path" in text


def test_safety_governor_and_maneuver_arbitration_are_wired():
    text = sim_source()
    assert "SafetyMotion.approachFactor" in text
    assert "SafetyMotion.chooseManeuver" in text
    assert "targetFactor" in text
    assert "appliedFactor" in text
    assert "sceneClearance >= SCALE.au(SAFE.C)" in text
    assert "SCALE.m(PERSON_R)" in text
    assert 'const sequenceFactor = s.k === "wait" ? 1 : SAFE.appliedFactor' in text
    assert 'const plannable = s && (s.k === "ik" || s.k === "ikOff" || s.k === "home")' in text
    assert "AVOID.plannedMotion = plannable" in text
    assert "AVOID.pathFactor" in text


def test_contact_monitor_unifies_main_and_extra_people_with_swept_links():
    text = sim_source()
    assert "function safetyPeople" in text
    assert "const ARM_SWEEP" in text
    assert "for (const target of safetyPeople())" in text
    assert "SafetyMotion.sweptSegmentCapsuleContact" in text
    assert 'kind:"main"' in text
    assert 'kind:"extra"' in text


def test_robot_contact_accidents_are_routed_to_run_tab():
    text = sim_source()
    run_tab = next(line for line in text.splitlines() if '{ id:"run"' in line)
    assert '"로봇 접촉 사고"' in run_tab


def test_status_bar_shows_selected_mode_and_applied_speed():
    text = sim_source()
    assert "const safetyModeLabels" in text
    assert "Math.round(SAFE.appliedFactor * 100)" in text
    assert 'SAFE_LIFT:"안전 자세"' in text
    assert "if (PHYS.estop)" in text
    assert '"비상정지 · 속도 0%"' in text
    assert 'const activeMode = AVOID.mode !== "PROCEED" ? AVOID.mode' in text
    assert ': (SAFE.stopped ? "STOP"' in text
