/**
 * Run tools/lattice-check.mjs over many IFC files and print one row each.
 *
 *   node tools/lattice-batch.mjs <file-or-folder> [...]
 */
import { execFileSync } from "node:child_process";
import { readdirSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const files = process.argv.slice(2).flatMap((p) =>
  statSync(p).isDirectory()
    ? readdirSync(p)
        .filter((f) => f.toLowerCase().endsWith(".ifc"))
        .map((f) => path.join(p, f))
        .sort((a, b) => statSync(a).size - statSync(b).size)
    : [p],
);

const rows = [];
for (const file of files) {
  try {
    const out = execFileSync(process.execPath, [path.join(HERE, "lattice-check.mjs"), file, "--json"], {
      encoding: "utf8",
      maxBuffer: 64 * 1024 * 1024,
    });
    const r = JSON.parse(out.trim().split("\n").pop());
    const pct = (v) => `${Math.round(v * 100)}%`;
    rows.push({
      model: r.file.slice(0, 30),
      floors: r.meshes.floors,
      stairs: r.meshes.stairs,
      doors: r.meshes.doors,
      walls: r.meshes.walls,
      nodes: r.stats.nodes,
      stairN: r.stats.stairNodes,
      doorN: r.stats.doorNodes,
      pruned: r.stats.prunedNodes,
      angle: +r.stats.angleDeg.toFixed(1),
      parts: r.withWalls.parts,
      largest: pct(r.withWalls.largestShare),
      second: r.parts.length > 1 ? pct(r.parts[1].nodes / r.stats.nodes) : "-",
      ms: Math.round(r.stats.ms),
    });
  } catch (error) {
    rows.push({ model: path.basename(file).slice(0, 30), floors: `ERROR ${String(error.message).slice(0, 80)}` });
  }
}
console.table(rows);
