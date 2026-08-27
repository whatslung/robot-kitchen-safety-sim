(function attachSafetyMotion(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.SafetyMotion = api;
})(typeof window !== "undefined" ? window : this, function buildSafetyMotion() {
  "use strict";

  function isPoint(point) {
    return point
      && Number.isFinite(point.x)
      && Number.isFinite(point.y)
      && Number.isFinite(point.z);
  }

  function copyPoint(point) {
    return { x: point.x, y: point.y, z: point.z };
  }

  function subtract(a, b) {
    return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
  }

  function dot(a, b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
  }

  function interpolate(a, b, amount) {
    return {
      x: a.x + (b.x - a.x) * amount,
      y: a.y + (b.y - a.y) * amount,
      z: a.z + (b.z - a.z) * amount,
    };
  }

  function segmentSegmentDistance(a0, a1, b0, b1) {
    const u = subtract(a1, a0);
    const v = subtract(b1, b0);
    const w = subtract(a0, b0);
    const aa = dot(u, u);
    const bb = dot(u, v);
    const cc = dot(v, v);
    const dd = dot(u, w);
    const ee = dot(v, w);
    const epsilon = 1e-12;
    let numeratorS;
    let denominatorS = aa * cc - bb * bb;
    let numeratorT;
    let denominatorT = denominatorS;

    if (denominatorS < epsilon) {
      numeratorS = 0;
      denominatorS = 1;
      numeratorT = ee;
      denominatorT = cc;
    } else {
      numeratorS = bb * ee - cc * dd;
      numeratorT = aa * ee - bb * dd;
      if (numeratorS < 0) {
        numeratorS = 0;
        numeratorT = ee;
        denominatorT = cc;
      } else if (numeratorS > denominatorS) {
        numeratorS = denominatorS;
        numeratorT = ee + bb;
        denominatorT = cc;
      }
    }

    if (numeratorT < 0) {
      numeratorT = 0;
      if (-dd < 0) numeratorS = 0;
      else if (-dd > aa) numeratorS = denominatorS;
      else {
        numeratorS = -dd;
        denominatorS = aa;
      }
    } else if (numeratorT > denominatorT) {
      numeratorT = denominatorT;
      if (-dd + bb < 0) numeratorS = 0;
      else if (-dd + bb > aa) numeratorS = denominatorS;
      else {
        numeratorS = -dd + bb;
        denominatorS = aa;
      }
    }

    const s = Math.abs(numeratorS) < epsilon ? 0 : numeratorS / denominatorS;
    const t = Math.abs(numeratorT) < epsilon ? 0 : numeratorT / denominatorT;
    const separation = {
      x: w.x + s * u.x - t * v.x,
      y: w.y + s * u.y - t * v.y,
      z: w.z + s * u.z - t * v.z,
    };
    return Math.hypot(separation.x, separation.y, separation.z);
  }

  function samplePlannedPath(current, path, pathIndex, speed, sampleTimes) {
    if (!isPoint(current)) return [];
    const times = Array.isArray(sampleTimes) ? sampleTimes : [];
    const stationary = () => times.map(() => copyPoint(current));
    if (!Array.isArray(path) || !Number.isFinite(pathIndex)
        || !Number.isFinite(speed) || speed <= 0) return stationary();

    const startIndex = Math.max(0, Math.floor(pathIndex));
    const remaining = path.slice(startIndex);
    if (!remaining.length || remaining.some(point => !isPoint(point))) return stationary();

    const points = [copyPoint(current), ...remaining.map(copyPoint)];
    const segments = [];
    let totalLength = 0;
    for (let i = 1; i < points.length; i += 1) {
      const from = points[i - 1];
      const to = points[i];
      const dx = to.x - from.x;
      const dy = to.y - from.y;
      const dz = to.z - from.z;
      const length = Math.hypot(dx, dy, dz);
      if (length > 1e-9) {
        segments.push({ from, to, start: totalLength, length });
        totalLength += length;
      }
    }
    if (!segments.length) return stationary();

    return times.map(rawTime => {
      const time = Number.isFinite(rawTime) ? Math.max(0, rawTime) : 0;
      const distance = Math.min(totalLength, time * speed);
      let segment = segments[segments.length - 1];
      for (const candidate of segments) {
        if (distance <= candidate.start + candidate.length) {
          segment = candidate;
          break;
        }
      }
      const along = Math.min(segment.length, Math.max(0, distance - segment.start));
      const ratio = along / segment.length;
      return {
        x: segment.from.x + (segment.to.x - segment.from.x) * ratio,
        y: segment.from.y + (segment.to.y - segment.from.y) * ratio,
        z: segment.from.z + (segment.to.z - segment.from.z) * ratio,
      };
    });
  }

  function sweptSegmentCapsuleContact(options) {
    const previous = options && options.previous;
    const current = options && options.current;
    const capsule = options && options.capsule;
    const linkRadius = options && options.linkRadius;
    const valid = previous && current && capsule
      && isPoint(previous.a) && isPoint(previous.b)
      && isPoint(current.a) && isPoint(current.b)
      && isPoint(capsule.center)
      && Number.isFinite(linkRadius) && linkRadius >= 0
      && Number.isFinite(capsule.radius) && capsule.radius >= 0
      && Number.isFinite(capsule.halfHeight) && capsule.halfHeight >= capsule.radius;
    if (!valid) {
      return { hit: true, invalid: true, clearance: Number.NEGATIVE_INFINITY, time: 0 };
    }

    const configuredStep = options.maxStep;
    const maxStep = Number.isFinite(configuredStep) && configuredStep > 0
      ? configuredStep : 0.04;
    const endpointTravel = Math.max(
      Math.hypot(
        current.a.x - previous.a.x,
        current.a.y - previous.a.y,
        current.a.z - previous.a.z,
      ),
      Math.hypot(
        current.b.x - previous.b.x,
        current.b.y - previous.b.y,
        current.b.z - previous.b.z,
      ),
    );
    const steps = Math.max(1, Math.ceil(endpointTravel / maxStep));
    const axisHalfHeight = Math.max(0, capsule.halfHeight - capsule.radius);
    const capsuleBottom = {
      x: capsule.center.x,
      y: capsule.center.y - axisHalfHeight,
      z: capsule.center.z,
    };
    const capsuleTop = {
      x: capsule.center.x,
      y: capsule.center.y + axisHalfHeight,
      z: capsule.center.z,
    };
    const combinedRadius = linkRadius + capsule.radius;
    let minimumClearance = Number.POSITIVE_INFINITY;
    let firstHit = null;

    for (let index = 0; index <= steps; index += 1) {
      const time = index / steps;
      const a = interpolate(previous.a, current.a, time);
      const b = interpolate(previous.b, current.b, time);
      const clearance = segmentSegmentDistance(a, b, capsuleBottom, capsuleTop)
        - combinedRadius;
      minimumClearance = Math.min(minimumClearance, clearance);
      if (firstHit === null && clearance <= 0) firstHit = time;
    }

    return {
      hit: firstHit !== null,
      invalid: false,
      clearance: minimumClearance,
      time: firstHit === null ? null : firstHit,
    };
  }

  function clampUnit(value) {
    return Math.min(1, Math.max(0, value));
  }

  function approachFactor(current, target, dtSec, decelRate, accelRate, immediateStop) {
    const values = [current, target, dtSec, decelRate, accelRate];
    if (values.some(value => !Number.isFinite(value))
        || dtSec < 0 || decelRate < 0 || accelRate < 0) return 0;
    const from = clampUnit(current);
    const to = clampUnit(target);
    if (immediateStop && to === 0) return 0;
    if (to < from) return Math.max(to, from - decelRate * dtSec);
    return Math.min(to, from + accelRate * dtSec);
  }

  function validCandidate(candidate) {
    return candidate
      && typeof candidate.safe === "boolean"
      && (Number.isFinite(candidate.clearance)
          || candidate.clearance === Number.POSITIVE_INFINITY);
  }

  function chooseManeuver(options) {
    const settings = options || {};
    const order = ["PROCEED", "HOLD", "RETRACT", "SAFE_LIFT", "STOP"];
    const currentMode = order.includes(settings.currentMode)
      ? settings.currentMode : "PROCEED";
    let chosen;

    if (settings.danger === false) {
      chosen = validCandidate(settings.proceed) && settings.proceed.safe
        ? { mode: "PROCEED", reason: "path-clear", clearance: settings.proceed.clearance }
        : { mode: "STOP", reason: "invalid-or-blocked-proceed", clearance: null };
    } else if (settings.danger === true && settings.beforeCross === true) {
      chosen = { mode: "HOLD", reason: "danger-before-cross", clearance: null };
    } else if (settings.danger === true && settings.beforeCross === false) {
      if (validCandidate(settings.retract) && settings.retract.safe) {
        chosen = {
          mode: "RETRACT", reason: "safe-retract", clearance: settings.retract.clearance,
        };
      } else if (validCandidate(settings.safeLift) && settings.safeLift.safe) {
        chosen = {
          mode: "SAFE_LIFT", reason: "retract-blocked", clearance: settings.safeLift.clearance,
        };
      } else {
        chosen = { mode: "STOP", reason: "no-safe-candidate", clearance: null };
      }
    } else {
      chosen = { mode: "STOP", reason: "invalid-danger-state", clearance: null };
    }

    const elapsed = Number.isFinite(settings.holdMs) ? Math.max(0, settings.holdMs) : 0;
    const minimum = Number.isFinite(settings.minHoldMs) ? Math.max(0, settings.minHoldMs) : 300;
    const easingRisk = order.indexOf(chosen.mode) < order.indexOf(currentMode);
    if (easingRisk && elapsed < minimum) {
      return { mode: currentMode, reason: "minimum-hold", clearance: chosen.clearance };
    }
    return chosen;
  }

  function trajectoryClearance(robotPoints, people, robotRadius, personRadius) {
    if (!Array.isArray(robotPoints) || robotPoints.length === 0
        || robotPoints.some(point => !isPoint(point))
        || !Array.isArray(people)
        || !Number.isFinite(robotRadius) || robotRadius < 0
        || !Number.isFinite(personRadius) || personRadius < 0) {
      return Number.NEGATIVE_INFINITY;
    }
    let clearance = Number.POSITIVE_INFINITY;
    for (const personPath of people) {
      if (!personPath || !Array.isArray(personPath.points)
          || personPath.points.length < robotPoints.length
          || personPath.points.some(point => !isPoint(point))) {
        return Number.NEGATIVE_INFINITY;
      }
      for (let index = 0; index < robotPoints.length; index += 1) {
        const robot = robotPoints[index];
        const personPoint = personPath.points[index];
        const horizontalDistance = Math.hypot(
          robot.x - personPoint.x,
          robot.z - personPoint.z,
        );
        clearance = Math.min(clearance, horizontalDistance - robotRadius - personRadius);
      }
    }
    return clearance;
  }

  function armTrajectoryClearance(armSamples, people, linkRadius) {
    if (!Array.isArray(armSamples) || armSamples.length === 0
        || !Array.isArray(people)
        || !Number.isFinite(linkRadius) || linkRadius < 0) {
      return Number.NEGATIVE_INFINITY;
    }
    let clearance = Number.POSITIVE_INFINITY;
    for (let sampleIndex = 0; sampleIndex < armSamples.length; sampleIndex += 1) {
      const links = armSamples[sampleIndex];
      if (!Array.isArray(links) || links.length === 0
          || links.some(link => !link || !isPoint(link.a) || !isPoint(link.b))) {
        return Number.NEGATIVE_INFINITY;
      }
      for (const personPath of people) {
        const personPoint = personPath && personPath.points
          && personPath.points[sampleIndex];
        const radius = personPath && personPath.radius;
        const halfHeight = personPath && personPath.halfHeight;
        if (!isPoint(personPoint)
            || !Number.isFinite(radius) || radius < 0
            || !Number.isFinite(halfHeight) || halfHeight < radius) {
          return Number.NEGATIVE_INFINITY;
        }
        const axisHalfHeight = Math.max(0, halfHeight - radius);
        const bottom = {
          x: personPoint.x,
          y: personPoint.y - axisHalfHeight,
          z: personPoint.z,
        };
        const top = {
          x: personPoint.x,
          y: personPoint.y + axisHalfHeight,
          z: personPoint.z,
        };
        for (const link of links) {
          clearance = Math.min(
            clearance,
            segmentSegmentDistance(link.a, link.b, bottom, top) - linkRadius - radius,
          );
        }
      }
    }
    return clearance;
  }

  return {
    samplePlannedPath,
    segmentSegmentDistance,
    sweptSegmentCapsuleContact,
    approachFactor,
    chooseManeuver,
    trajectoryClearance,
    armTrajectoryClearance,
  };
});
