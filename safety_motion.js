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

  return { samplePlannedPath, segmentSegmentDistance, sweptSegmentCapsuleContact };
});
