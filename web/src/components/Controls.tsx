import { type ReactNode } from "react";
import {
  thetaMaxOf,
  zoneColor,
  type CameraModel,
  type Params,
  type ParamSpec,
  type Vec3,
} from "../lib/cameras";

interface Props {
  model: CameraModel;
  params: Params;
  onParams: (patch: Params) => void;
  active: Vec3;
}

const RES_PRESETS = [
  { name: "VGA", w: 640, h: 480 },
  { name: "Square", w: 720, h: 720 },
  { name: "720p", w: 1280, h: 720 },
];

function fmt(v: number, step: number) {
  if (step >= 1) return v.toFixed(0);
  if (step >= 0.01) return v.toFixed(2);
  if (step >= 0.001) return v.toFixed(3);
  return v.toExponential(1);
}

export function Controls({ model, params, onParams, active }: Props) {
  const proj = model.project(active, params);
  const tmaxDeg = (thetaMaxOf(model, params) * 180) / Math.PI;
  const W = Math.round(params.cx * 2);
  const H = Math.round(params.cy * 2);

  const setSpec = (spec: ParamSpec, v: number) => {
    const patch: Params = {};
    for (const key of spec.link ?? [spec.key]) patch[key] = v;
    onParams(patch);
  };

  return (
    <div className="flex flex-col gap-6">
      <Section title={`${model.name} — parameters`}>
        <div className="flex flex-col gap-4">
          {model.params.map((spec) => (
            <Slider
              key={spec.key}
              symbol={spec.symbol}
              label={spec.label}
              value={params[spec.key]}
              min={spec.min}
              max={spec.max}
              step={spec.step}
              onChange={(v) => setSpec(spec, v)}
              format={(v) => fmt(v, spec.step)}
            />
          ))}
        </div>
        <p className="mt-3 text-[11px] leading-snug text-faint">{model.blurb}</p>
      </Section>

      <Section title="Image resolution">
        <div className="flex flex-col gap-4">
          <Slider symbol="W" label="sensor width" value={W} min={320} max={1920} step={16}
            onChange={(w) => onParams({ cx: w / 2 })} format={(v) => `${v.toFixed(0)}px`} />
          <Slider symbol="H" label="sensor height" value={H} min={240} max={1440} step={16}
            onChange={(h) => onParams({ cy: h / 2 })} format={(v) => `${v.toFixed(0)}px`} />
        </div>
        <div className="grid grid-cols-3 gap-2 mt-3">
          {RES_PRESETS.map((r) => (
            <PresetBtn key={r.name} title={r.name} sub={`${r.w}×${r.h}`}
              onClick={() => onParams({ cx: r.w / 2, cy: r.h / 2 })} />
          ))}
        </div>
      </Section>

      <Section title="Inside the model — picked ray">
        <dl className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-line bg-line">
          <Stat label="incidence θ" value={`${((proj.theta * 180) / Math.PI).toFixed(1)}°`} />
          <Stat label="zone" value={proj.zone} color={zoneColor[proj.zone]} />
          <Stat label="pixel u" value={proj.valid ? proj.u.toFixed(1) : "—"} />
          <Stat label="pixel v" value={proj.valid ? proj.v.toFixed(1) : "—"} />
          <Stat label="captured?" value={proj.valid ? "yes" : "off-image"}
            color={proj.valid ? "var(--color-valid)" : "var(--color-invalid)"} />
          <Stat label="field of view" value={`${(tmaxDeg * 2).toFixed(0)}°`} />
        </dl>
      </Section>
    </div>
  );
}

interface SliderProps {
  label: string;
  symbol: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format: (v: number) => string;
}

function Slider({ label, symbol, value, min, max, step, onChange, format }: SliderProps) {
  return (
    <label className="block">
      <div className="flex items-baseline justify-between mb-1.5">
        <span className="text-[11px] uppercase tracking-[0.08em] text-muted">
          <span className="text-text font-mono">{symbol}</span> · {label}
        </span>
        <span className="font-mono text-sm text-primary tnum">{format(value)}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))} />
    </label>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section>
      <h3 className="text-[11px] uppercase tracking-[0.14em] text-faint mb-3">{title}</h3>
      {children}
    </section>
  );
}

function PresetBtn({ title, sub, onClick }: { title: string; sub: string; onClick: () => void }) {
  return (
    <button onClick={onClick}
      className="rounded-md border border-line bg-panel px-2 py-2 text-left transition hover:border-primary hover:bg-panel-2">
      <div className="text-[12px] text-text leading-tight">{title}</div>
      <div className="font-mono text-[11px] text-muted tnum">{sub}</div>
    </button>
  );
}

function Stat({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="bg-panel px-3 py-2.5">
      <dt className="text-[10px] uppercase tracking-[0.06em] text-faint">{label}</dt>
      <dd className="font-mono text-sm tnum" style={{ color: color ?? "var(--color-text)" }}>
        {value}
      </dd>
    </div>
  );
}
