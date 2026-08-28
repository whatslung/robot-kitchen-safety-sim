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


def test_dedicated_demo_uses_only_fresh_lstm_occupancy():
    text = sim_source()
    start = text.index("function learnedPeopleOccupancy")
    end = text.index("function plannedMinimumBaseDistance", start)
    learned = text[start:end]
    assert 'MPRED.pred.get("gt:0")' in learned
    assert "SafetyMotion.predictionFresh" in learned
    assert "modePosAt" in learned
    assert "modeSigAt" in learned
    assert "plannedPeopleOccupancy" not in learned
    assert "person.path" not in learned
    assert 'AVOID.predictionSource === "lstm"' in text
    assert 'predictionSource:"planned"' in text
    assert "AVOID.predictionReady" in text


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
    assert "Math.min(SAFE.targetFactor, AVOID.pathFactor)" in text
    assert "SafetyMotion.motionStopRequired" in text
    assert "AVOID.escapeVerified" in text
    assert "SafetyMotion.keepVerifiedSafeLift" in text
    assert "effectiveTargetFactor" in text
    assert "SAFE.stopped || AVOID.targetFactor === 0" not in text
    assert "AVOID.clearSince" in text
    assert "clearMs" in text


def test_manual_velocity_tracking_is_independent_of_visualization():
    text = sim_source()
    assert "function peopleVelocityUpdate" in text
    assert "peopleVelocityUpdate(dt / 1000)" in text


def test_safe_lift_fails_closed_and_replans_every_exit():
    text = sim_source()
    assert "checkEnvironment && (!PHYS.ready || !PHYS.links.length)" in text
    assert "originalFrom:state.stepStartJoints.slice()" in text
    assert 'AVOID.mode === "RETRACT"' in text
    assert "state.seqT = state.dur" in text
    assert "retractPrepared" in text
    assert "AVOID.lift.originalTo" in text
    assert 'AVOID.mode = "STOP"' in text
    assert "clearMsByMode" in text


def test_contact_monitor_unifies_main_and_extra_people_with_swept_links():
    text = sim_source()
    assert "function safetyPeople" in text
    assert "const ARM_SWEEP" in text
    assert "const targets = safetyPeople();" in text
    assert "for (const target of targets)" in text
    assert "SafetyMotion.sweptSegmentCapsuleContact" in text
    assert 'kind:"main"' in text
    assert 'kind:"extra"' in text


def test_held_basket_is_part_of_candidate_and_swept_contact_geometry():
    text = sim_source()
    assert "function basketPayloadSegments" in text
    assert "if (state.basketHeld) links.push(...basketPayloadSegments())" in text
    assert "segment.radius ?? ARM_R" in text
    assert "link.radius ?? linkRadius" in (Path(__file__).parents[1] / "safety_motion.js").read_text(encoding="utf-8")


def test_lstm_yield_demo_starts_robot_before_the_person_and_restores_route_blockers():
    text = sim_source()
    assert 'id="lstmYieldBtn"' in text
    assert "function startAutoWork" in text
    assert "startAutoWork();" in text
    assert "stepUpdate(16);" in text
    assert "personStartDueAt:0" in text
    assert "personStartDueAt:now" in text
    assert "LSTM_YIELD_DEMO.startedAt + 0" in text
    assert "observationCount >= 8" in text
    assert "function lstmYieldDemoRoute" in text
    assert "function disableBlockingDemoProps" in text
    assert 'const removableFiles = new Set(["env_pancart.glb"])' in text
    assert "const body = PUSH.bodies.find(item => item.idx === index)" in text
    assert "idx:it.idx" in text
    assert "linearVelocity:body.ag.body.getLinearVelocity().clone()" in text
    assert "angularVelocity:body.ag.body.getAngularVelocity().clone()" in text
    assert "disablePreStep:body.ag.body.disablePreStep" in text
    assert "buildPersonColliders();" in text
    assert "buildNavGrid();" in text
    assert "function restoreDemoRouteBlockers" in text
    assert "function prepareLstmDemoCrossingPath" in text
    assert "disableBlockingDemoProps(intendedRoute, true)" in text
    assert "const crossingPath = prepareLstmDemoCrossingPath" in text
    assert 'AVOID.predictionSource = "lstm"' in text
    assert 'AVOID.predictionSource = "current-only"' in text
    assert "lstmYieldDemoUpdate(now, dt);" in text
    assert 'AVOID.predictionSource === "lstm" ? 2.0 : 0.4' in text
    assert "learnedRecord.risk.tEntryStop" in text
    assert "const danger = !proceed.safe || learnedRisk" in text
    assert 'AVOID.mode === "RETRACT" ? 0.65' in text
    assert 'AVOID.mode === "SAFE_LIFT" ? 0.60' in text
    assert "const complete = home && D.basketDelivered" in text
    assert "MPRED.reqId++;" in text
    assert "record.requestAt >= D.personStartedAt + 7 * 400" in text
    assert "const emergencyStop = PHYS.estop" in text
    assert 'const blockedByStop = PHYS.estop || (SAFE.stopped && mode === "PROCEED")' in text
    assert "const danger = !proceed.safe || learnedRisk || SAFE.stopped" in text
    assert "basketNode.getChildMeshes(false)" in text
    assert "if (!contact) continue;" in text
    assert "if (!moving || !contact) continue;" not in text
    assert "&& !SAFE.stopped" not in text
    assert 'finishLstmYieldDemo("cancelled", null, true)' in text
    assert "function resetArmSweepBaseline" in text
    assert 'finishLstmYieldDemo("failed", "LSTM 예측 중단' in text
    assert "function demoSafeCrossingPath" in text
    assert "function currentSafetyPeopleOccupancy(times)" in text
    assert "for (const target of safetyPeople())" in text
    assert "learned.concat(currentSafetyPeopleOccupancy(times))" in text
    assert "SAFE.NOM_SLOW+0.25" in text
    assert "SAFE.NOM_STOP-0.25" in text
    assert "enteredStopRadius:false" in text
    assert "escapeMotionInsideStop:false" in text
    assert "D.minDistance = Math.min(D.minDistance, SAFE.dist)" in text
    assert "const personOutsideSlowRadius = SAFE.dist >= SAFE.slowR" in text
    assert "&& personOutsideSlowRadius" in text


def test_directional_collision_and_avoidance_controls_share_the_same_a_to_b_route():
    text = sim_source()
    assert 'id="collisionScenarioBtn"' in text
    assert 'id="avoidanceScenarioBtn"' in text
    assert "function directionalScenarioRoute" in text
    assert 'startDirectionalScenario("collision")' in text
    assert "const route = directionalScenarioRoute();" in text
    assert "AVOID.safeLiftTarget = T.rightSafe" in text
    assert "function startLstmYieldDemo(routeOverride)" in text
    assert "route.directionalPath" in text
    assert "return route.directionalPath.map(point => point.clone());" in text
    assert "contactPoint" in text
    assert "person.node.position.copyFrom(route.start);" in text
    assert "personGo(route.points.map(point => point.clone()), { speed:0.72 })" in text
    assert "personFall();" in text


def test_demo_removes_tray_carts_before_the_a_to_b_route_is_followed():
    text = sim_source()
    start = text.index("function disableBlockingDemoProps")
    end = text.index("function prepareLstmDemoCrossingPath", start)
    body = text[start:end]
    assert 'if (file === "env_pancart.glb")' in body
    assert body.index('if (file === "env_pancart.glb")') < body.index("if (!bounds || !samples.some")
    assert 'if (file === "env_rack.glb")' not in body


def test_lstm_demo_starts_at_the_robot_front_left_diagonal():
    text = sim_source()
    start = text.index("function lstmYieldDemoRoute")
    end = text.index("function demoRouteSamples", start)
    route = text[start:end]
    assert "B.x+1.25" in route
    assert "B.z+3.85" in route
    assert "const demoGate = new V3(B.x+1.10, 0, B.z+3.75)" in route
    assert "gate:demoGate.clone()" in route


def test_a_to_b_scenarios_emit_coordinate_and_arm_debug_traces():
    text = sim_source()
    assert "const SCENARIO_TRACE" in text
    assert "function scenarioTraceUpdate" in text
    assert 'console.info("[A2B]", record)' in text
    assert "person:{ x:" in text
    assert "tcp:{ x:" in text
    assert "joints:JOINTS.map" in text
    assert "scenarioTraceUpdate(now);" in text
    assert "get scenarioTrace() { return SCENARIO_TRACE; }" in text


def test_safe_lift_candidate_is_scored_even_before_cross_progress():
    text = sim_source()
    start = text.index("let retract =", text.index("function avoidDecide"))
    end = text.index("const candidateByMode", start)
    candidates = text[start:end]
    assert "if (danger)" in candidates
    assert "if (progress > AVOID.startT)" in candidates
    assert candidates.index("if (progress > AVOID.startT)") < candidates.index("AVOID.safeLiftTarget || T.potHover")
    assert 'retract = scoreArmCandidate(state.stepStartJoints, occupancy, true)' in candidates


def test_normal_auto_button_uses_shared_start_without_enabling_lstm_source():
    text = sim_source()
    handler = text[text.index('$("autoBtn").addEventListener'):]
    handler = handler[:handler.index('$("stopBtn").addEventListener')]
    assert "startAutoWork()" in handler
    assert 'predictionSource = "lstm"' not in handler


def test_robot_contact_accidents_are_routed_to_run_tab():
    text = sim_source()
    run_tab = next(line for line in text.splitlines() if '{ id:"run"' in line)
    assert '"로봇 접촉 사고"' in run_tab


def test_status_bar_shows_selected_mode_and_applied_speed():
    text = sim_source()
    assert "const safetyModeLabels" in text
    assert "Math.round(SAFE.appliedFactor * 100)" in text
    assert 'SAFE_LIFT:"안전 자세 회피"' in text
    assert "if (PHYS.estop)" in text
    assert '"비상정지 · 속도 0%"' in text
    assert 'const activeMode = invalidEscape ? "STOP"' in text
    assert 'const invalidEscape = (AVOID.mode === "RETRACT" || AVOID.mode === "SAFE_LIFT")' in text
    assert ': (SAFE.stopped ? "STOP"' in text
    assert 'const severe = activeMode === "STOP"' in text
    assert 'activeMode === "STOP" || SAFE.stopped' not in text
