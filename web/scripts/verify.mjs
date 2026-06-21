// Compare the TS camera ports against the ds_msp reference dump.
// Bundle first:  esbuild scripts/verify.mjs --bundle --format=esm --platform=node --outfile=/tmp/verify.bundle.mjs
import { readFileSync } from "node:fs";
import { CAMERA_BY_ID } from "../src/lib/cameras.ts";

const ref = JSON.parse(readFileSync("/tmp/ds_ref.json", "utf8"));

let worstPx = 0;
let worstRay = 0;
let validMismatch = 0;
let nProj = 0;
let nRay = 0;
const perModel = {};

for (const [id, recs] of Object.entries(ref)) {
  const model = CAMERA_BY_ID[id];
  let mPx = 0,
    mRay = 0,
    mVal = 0;
  for (const r of recs) {
    const pr = model.project(r.P, model.defaults);
    if (pr.valid !== r.valid) mVal++;
    if (r.valid && Number.isFinite(r.u)) {
      const dpx = Math.hypot(pr.u - r.u, pr.v - r.v);
      mPx = Math.max(mPx, dpx);
      nProj++;
    }
    if (r.ray) {
      const un = model.unproject(r.u, r.v, model.defaults);
      const dot = Math.abs(
        un.ray[0] * r.ray[0] + un.ray[1] * r.ray[1] + un.ray[2] * r.ray[2],
      );
      const ang = Math.acos(Math.min(1, dot));
      mRay = Math.max(mRay, ang);
      nRay++;
    }
  }
  perModel[id] = { px: mPx, rayRad: mRay, validMismatch: mVal };
  worstPx = Math.max(worstPx, mPx);
  worstRay = Math.max(worstRay, mRay);
  validMismatch += mVal;
}

console.log("per-model:");
for (const [id, m] of Object.entries(perModel)) {
  console.log(
    `  ${id.padEnd(7)}  proj Δ=${m.px.toExponential(2)} px   unproj Δ=${m.rayRad.toExponential(2)} rad   valid-mismatch=${m.validMismatch}`,
  );
}
console.log(
  `\nWORST  proj Δ=${worstPx.toExponential(3)} px over ${nProj} pts | unproj Δ=${worstRay.toExponential(3)} rad over ${nRay} pts | valid mismatches=${validMismatch}`,
);
// gate tight enough to actually defend the ~1e-12 px / ~1e-8 rad claim
const pass = worstPx < 1e-11 && worstRay < 1e-7 && validMismatch === 0;
console.log(pass ? "VERIFY PASS ✓" : "VERIFY FAIL ✗");
process.exit(pass ? 0 : 1);
