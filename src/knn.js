/**
 * knn.js
 * K-Nearest Neighbour classifier over PCA manifold centroids.
 * Works entirely in the browser — no server needed.
 *
 * buildIndex(speciesMap)  → index (array of { species, centroid })
 * classify(manifold, index, k)  → top-k [{ rank, species, distance }]
 * distToSimilarity(distance)  → similarity percentage (0–100)
 */

/**
 * Compute the centroid (mean xyz) of a manifold's trajectory.
 * @param {object} manifold  — { xyz: [[x,y,z], ...] }
 * @returns {[number, number, number]}
 */
function centroid(manifold) {
  const pts = manifold.xyz;
  const n   = pts.length;
  if (!n) return [0, 0, 0];
  let sx = 0, sy = 0, sz = 0;
  for (const [x, y, z] of pts) { sx += x; sy += y; sz += z; }
  return [sx / n, sy / n, sz / n];
}

/**
 * Euclidean distance between two 3-vectors.
 */
function dist3(a, b) {
  const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * buildIndex(speciesMap)
 * Pre-computes the centroid for every species in the loaded data.
 * Call once after birdsong_data.json is fetched.
 *
 * @param {object} speciesMap  — { speciesKey: { xyz, t, energy, ... } }
 * @returns {Array<{ species: string, centroid: [number,number,number] }>}
 */
export function buildIndex(speciesMap) {
  return Object.entries(speciesMap).map(([species, manifold]) => ({
    species,
    centroid: centroid(manifold),
  }));
}

/**
 * classify(manifold, index, k)
 * Finds the k closest species to the given manifold by centroid distance.
 *
 * @param {object} manifold  — { xyz: [[x,y,z], ...] }
 * @param {Array}  index     — output of buildIndex()
 * @param {number} k         — number of results to return (default 3)
 * @returns {Array<{ rank, species, distance }>}
 */
export function classify(manifold, index, k = 3) {
  const c = centroid(manifold);
  return index
    .map(entry => ({
      species:  entry.species,
      distance: dist3(c, entry.centroid),
    }))
    .sort((a, b) => a.distance - b.distance)
    .slice(0, k)
    .map((entry, i) => ({ rank: i + 1, ...entry }));
}

/**
 * distToSimilarity(distance)
 * Maps a Euclidean distance in PCA space to a 0–100% similarity score.
 * Uses an exponential decay: similarity = 100 * exp(-k * distance)
 * Calibrated so distance 0 → 100%, distance 2 → ~10%.
 */
export function distToSimilarity(distance) {
  const score = 100 * Math.exp(-1.15 * distance);
  return Math.round(Math.max(1, Math.min(100, score)));
}
