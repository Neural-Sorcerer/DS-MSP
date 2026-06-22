import { CAMERAS, type CameraModel } from "../lib/cameras";

function Glyph({ d }: { d: string }) {
  return (
    <svg viewBox="0 0 24 24" className="w-6 h-6" aria-hidden>
      <path d={d} fill="none" stroke="currentColor" strokeWidth={1.4} strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

interface Props {
  selectedId: string;
  onSelect: (m: CameraModel) => void;
}

// The camera rail: every model the library actually ships, each with its own
// mark. Picking one swaps the parameter menu and the whole visualisation.
export function ModelSelector({ selectedId, onSelect }: Props) {
  return (
    <div className="grid grid-cols-3 sm:grid-cols-6 gap-2">
      {CAMERAS.map((m) => {
        const on = m.id === selectedId;
        return (
          <button
            key={m.id}
            onClick={() => onSelect(m)}
            title={m.blurb}
            aria-pressed={on}
            className={[
              "group flex flex-col items-center gap-1.5 rounded-lg border px-2 py-3 transition",
              on
                ? "border-primary bg-primary/10 text-primary"
                : "border-line bg-panel text-muted hover:border-primary/60 hover:text-text",
            ].join(" ")}
          >
            <Glyph d={m.glyph} />
            <span className="text-[11px] leading-tight text-center font-medium">
              {m.name}
            </span>
            {on && (
              <span className="text-[9px] uppercase tracking-[0.1em] text-primary/80">
                {m.wideFov ? "wide FOV" : "narrow"}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}
