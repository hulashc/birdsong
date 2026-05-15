// knn.js — K-Nearest Neighbour classifier over PCA manifold centroids.
//
// Each species is represented as a single centroid: the mean of all its
// XYZ trajectory points in the 3D PCA space.  When an uploaded audio file
// is embedded, we compare its centroid against every known species centroid
// and return the top-k closest matches with Euclidean distances.
//
// No external dependencies — pure JS.

// ── Compute the mean XYZ point of a trajectory ────────────────────────────────
function centroid(xyz) {
  const n = xyz.length;
  if (n === 0) return [0, 0, 0];
  let sx = 0, sy = 0, sz = 0;
  for (const [x, y, z] of xyz) { sx += x; sy += y; sz += z; }
  return [sx / n, sy / n, sz / n];
}

// ── Euclidean distance between two 3-vectors ──────────────────────────────────
function dist3(a, b) {
  return Math.sqrt(
    (a[0] - b[0]) ** 2 +
    (a[1] - b[1]) ** 2 +
    (a[2] - b[2]) ** 2
  );
}

// ── Build reference centroids from birdsong_data.json structure ───────────────
// data: the full parsed JSON object  { robin: { xyz: [[x,y,z], ...], ... }, ... }
// Returns: Map<species, [cx, cy, cz]>
export function buildIndex(data) {
  const index = new Map();
  for (const [species, entry] of Object.entries(data)) {
    if (!entry.xyz || entry.xyz.length === 0) continue;
    index.set(species, centroid(entry.xyz));
  }
  return index;
}

// ── Classify an uploaded manifold against the index ───────────────────────────
// manifold : the object returned by extractManifold() — needs .xyz
// index    : Map returned by buildIndex()
// k        : how many nearest neighbours to return (default 3)
//
// Returns an array of k objects, sorted nearest-first:
//   [{ species: 'robin', distance: 0.142, rank: 1 }, ...]
export function classify(manifold, index, k = 3) {
  if (!manifold?.xyz || manifold.xyz.length === 0) return [];

  const queryCentroid = centroid(manifold.xyz);

  const results = [];
  for (const [species, c] of index) {
    results.push({ species, distance: dist3(queryCentroid, c) });
  }

  results.sort((a, b) => a.distance - b.distance);

  return results.slice(0, k).map((r, i) => ({ ...r, rank: i + 1 }));
}

// ── Format a distance as a human-readable similarity % ────────────────────────
// Distances in normalised PCA space range from ~0 (identical) to ~2 (opposite
// ends of the unit cube).  We map 0→100% and 2→0% linearly.
export function distToSimilarity(d) {
  return Math.max(0, Math.round((1 - d / 2) * 100));
}
