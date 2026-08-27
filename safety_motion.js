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

  return { samplePlannedPath };
});
