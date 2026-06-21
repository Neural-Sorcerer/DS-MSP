import * as THREE from "three";
import type { CameraModel, Params } from "./cameras";
import { computeRayGrid, type RayGrid } from "./raymap";

/** A float DataTexture holding an n×n ray grid, ready as a shader uniform. */
export function createRayTexture(n: number): THREE.DataTexture {
  const tex = new THREE.DataTexture(
    new Float32Array(n * n * 4),
    n,
    n,
    THREE.RGBAFormat,
    THREE.FloatType,
  );
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.needsUpdate = true;
  return tex;
}

export function writeRayTexture(tex: THREE.DataTexture, grid: RayGrid) {
  (tex.image.data as unknown as Float32Array).set(grid.data);
  tex.needsUpdate = true;
}

export function refreshRayTexture(
  tex: THREE.DataTexture,
  model: CameraModel,
  p: Params,
  n: number,
) {
  writeRayTexture(tex, computeRayGrid(model, p, n));
}
