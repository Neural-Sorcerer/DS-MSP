// Model-agnostic raw-image shader. It does not know any camera math — it reads a
// precomputed ray grid (a DataTexture filled by the verified unproject in
// raymap.ts) and, for each output pixel, samples the panorama along that pixel's
// bearing. So the same shader renders every camera model correctly, and matches
// the wrapped surfaces and the 3D background pixel-for-pixel.

export const sampleVertFullscreen = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;

export const sampleVertScene = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

export const sampleFrag = /* glsl */ `
  precision highp float;
  varying vec2 vUv;
  uniform sampler2D uRayTex;  // [rx,ry,rz,valid]
  uniform sampler2D uPano;    // equirectangular panorama
  const float PI = 3.141592653589793;

  vec2 equirectUv(vec3 d) {
    float u = atan(d.z, d.x) / (2.0 * PI) + 0.5;
    float v = asin(clamp(d.y, -1.0, 1.0)) / PI + 0.5;
    return vec2(u, v);
  }

  void main() {
    // ray grid row 0 is image-top (v=0); screen-top is vUv.y=1 → flip
    vec4 r = texture2D(uRayTex, vec2(vUv.x, 1.0 - vUv.y));
    if (r.w < 0.5) {
      gl_FragColor = vec4(0.043, 0.051, 0.078, 1.0); // outside FOV
      return;
    }
    vec3 dir = normalize(r.xyz);
    vec3 col = texture2D(uPano, equirectUv(dir)).rgb;
    float vig = smoothstep(1.05, 0.2, length(vUv - 0.5) * 1.7);
    gl_FragColor = vec4(col * (0.6 + 0.4 * vig), 1.0);
  }
`;
