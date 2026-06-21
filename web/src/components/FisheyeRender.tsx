import { Canvas } from "@react-three/fiber";
import { useTexture } from "@react-three/drei";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import type { CameraModel, Params } from "../lib/cameras";
import { createRayTexture, refreshRayTexture } from "../lib/rayTexture";
import { sampleFrag, sampleVertFullscreen } from "../lib/sampleShader";
import { ENV_URL } from "../lib/assets";

const GRID_N = 160;

function Quad({ model, params }: { model: CameraModel; params: Params }) {
  const pano = useTexture(ENV_URL);
  const rayTex = useMemo(() => createRayTexture(GRID_N), []);
  const matRef = useRef<THREE.ShaderMaterial>(null);

  useEffect(() => {
    pano.colorSpace = THREE.SRGBColorSpace;
    pano.minFilter = THREE.LinearMipmapLinearFilter;
    pano.anisotropy = 8;
    pano.needsUpdate = true;
  }, [pano]);

  // recompute the ray grid whenever the model or its params change
  useEffect(() => {
    refreshRayTexture(rayTex, model, params, GRID_N);
  }, [rayTex, model, params]);

  const uniforms = useMemo(
    () => ({ uRayTex: { value: rayTex }, uPano: { value: pano } }),
    [rayTex, pano],
  );

  return (
    <mesh>
      <planeGeometry args={[2, 2]} />
      <shaderMaterial
        ref={matRef}
        uniforms={uniforms}
        vertexShader={sampleVertFullscreen}
        fragmentShader={sampleFrag}
      />
    </mesh>
  );
}

export function FisheyeRender({ model, params }: { model: CameraModel; params: Params }) {
  return (
    <Canvas
      orthographic
      camera={{ position: [0, 0, 1], near: 0.1, far: 10 }}
      dpr={[1, 2]}
      gl={{ antialias: true }}
      style={{ width: "100%", height: "100%" }}
    >
      <Quad model={model} params={params} />
    </Canvas>
  );
}
