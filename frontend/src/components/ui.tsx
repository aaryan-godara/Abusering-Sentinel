import type { ReactNode } from 'react';
import clsx from 'clsx';

/* ==========================================================================
   Console primitives — floating glass panels, compact readouts, controls.
   Palette is strictly black / white / gray / red.
   ========================================================================== */

export function Panel({
  title,
  meta,
  actions,
  children,
  className,
  bodyClassName,
  strong = false,
}: {
  title?: string;
  meta?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
  strong?: boolean;
}) {
  return (
    <section
      className={clsx(
        'corner-ticks pointer-events-auto flex min-h-0 flex-col',
        strong ? 'floating-panel-strong' : 'floating-panel',
        className,
      )}
    >
      {(title || actions) && (
        <header className="flex shrink-0 items-center gap-2 border-b border-black/[0.1] px-3 py-2">
          {title && <h2 className="panel-title">{title}</h2>}
          {meta && <span className="panel-meta truncate">{meta}</span>}
          {actions && <div className="ml-auto flex items-center gap-1">{actions}</div>}
        </header>
      )}
      {/* pb-3 keeps content clear of the bottom-right corner tick */}
      <div className={clsx('min-h-0 flex-1 overflow-y-auto px-3 pb-3 pt-2', bodyClassName)}>
        {children}
      </div>
    </section>
  );
}

/** Tiny square glass icon button — used to collapse / reopen floating panels. */
export function IconButton({
  icon,
  label,
  onClick,
  active = false,
  className,
}: {
  icon: ReactNode;
  label: string;
  onClick: () => void;
  active?: boolean;
  className?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={label}
      aria-label={label}
      className={clsx(
        'glass-chip pointer-events-auto flex h-6 w-6 items-center justify-center',
        active ? 'text-signal-bright' : 'text-ink-50',
        className,
      )}
    >
      {icon}
    </button>
  );
}

/** Inline label / value line. */
export function Row({
  label,
  value,
  red = false,
  strong = false,
  wrap = false,
  mono = false,
}: {
  label: string;
  value: ReactNode;
  red?: boolean;
  strong?: boolean;
  wrap?: boolean;
  /** Render the value as a technical identifier (monospace). */
  mono?: boolean;
}) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-[4.5px]">
      <span className="label-caps shrink-0">{label}</span>
      <span
        className={clsx(
          'text-right',
          mono && 'tech-id',
          wrap ? 'break-all' : 'truncate',
          red || strong
            ? 'text-[14px] font-semibold leading-[1.4]'
            : 'text-[13px] font-medium leading-[1.5]',
          red ? 'text-signal-bright' : strong ? 'text-ink-50' : 'text-ink-100',
        )}
      >
        {value}
      </span>
    </div>
  );
}

/** Stacked label above a large value. */
export function Metric({
  label,
  value,
  red = false,
  unit,
  mono = true,
}: {
  label: string;
  value: string;
  red?: boolean;
  unit?: string;
  /** Numeric / identifier values use monospace; word values use sans. */
  mono?: boolean;
}) {
  return (
    <div>
      <div className="label-caps">{label}</div>
      <div
        className={clsx(
          'value-display mt-[3px]',
          mono && 'tech-num',
          red && 'text-signal-bright',
        )}
      >
        {value}
        {unit && <span className="ml-[1px] text-[13px] font-semibold">{unit}</span>}
      </div>
    </div>
  );
}

/** Thin magnitude bar. Gray = reduces risk, red = increases risk. */
export function Bar({ ratio, red }: { ratio: number; red: boolean }) {
  const width = `${Math.max(2, Math.min(100, ratio * 100))}%`;
  return (
    <div className="h-[3px] w-full rounded-full bg-black/[0.12]">
      <div
        className={clsx('h-full rounded-full', red ? 'bg-signal' : 'bg-ink-300')}
        style={{ width }}
      />
    </div>
  );
}

export function Tag({ children, red = false }: { children: ReactNode; red?: boolean }) {
  return (
    <span
      className={clsx(
        'rounded-md border px-1.5 py-[2.5px] text-[10.5px] font-semibold uppercase leading-[1.45] tracking-[0.045em]',
        red
          ? 'border-signal-dim bg-signal-deep/70 text-signal-bright'
          : 'border-ink-600 bg-white/70 text-ink-100',
      )}
    >
      {children}
    </span>
  );
}

/**
 * Qualitative evidence strength chip. HIGH is red because a directly shared
 * device or payment instrument is the strongest existing link type — it is not
 * a probability.
 */
export function StrengthChip({ strength }: { strength: 'HIGH' | 'MEDIUM' | 'INDIRECT' }) {
  return (
    <span
      className={clsx(
        'shrink-0 rounded border px-1 py-[1px] text-[9.5px] font-semibold uppercase leading-[1.4] tracking-[0.05em]',
        strength === 'HIGH'
          ? 'border-signal-dim bg-signal-deep/70 text-signal-bright'
          : strength === 'MEDIUM'
            ? 'border-ink-600 bg-white/70 text-ink-50'
            : 'border-ink-700 bg-white/60 text-ink-350',
      )}
    >
      {strength}
    </span>
  );
}

/** Compact readout used in the floating status/counter strips. */
export function StatChip({
  label,
  value,
  red = false,
}: {
  label: string;
  value: string;
  red?: boolean;
}) {
  return (
    <span className="flex shrink-0 items-baseline gap-1">
      <span className="label-caps">{label}</span>
      <span
        className={clsx(
          'tech-num text-[12.5px] font-semibold leading-[1.4]',
          red ? 'text-signal-bright' : 'text-ink-50',
        )}
      >
        {value}
      </span>
    </span>
  );
}

/** Small live/active pulse dot. */
export function LiveDot({ red = true }: { red?: boolean }) {
  return (
    <span
      className={clsx('status-dot', red ? 'status-dot-red' : 'status-dot-gray')}
      aria-hidden="true"
    />
  );
}

/** Numbered evidence line: "01  Shared Device — 4 accounts share a device." */
export function NumberedItem({
  index,
  title,
  detail,
  right,
}: {
  index: number;
  title: string;
  detail: string;
  right?: ReactNode;
}) {
  return (
    <div className="flex gap-2">
      <span className="tech-num shrink-0 text-[11px] font-semibold leading-[1.6] text-signal-bright">
        {String(index).padStart(2, '0')}
      </span>
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline gap-1.5">
          <span className="truncate text-[12.5px] font-semibold leading-[1.45] text-ink-50">
            {title}
          </span>
          {right && <span className="ml-auto">{right}</span>}
        </div>
        <p className="body-muted">{detail}</p>
      </div>
    </div>
  );
}

/** Compact control used in the floating control bar. */
export function ControlButton({
  icon,
  label,
  onClick,
  active = false,
  title,
}: {
  icon: ReactNode;
  label: string;
  onClick: () => void;
  active?: boolean;
  title?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title ?? label}
      aria-pressed={active}
      className={clsx(
        'flex items-center gap-1.5 rounded-md border px-2 py-1 text-[10.5px] font-semibold uppercase leading-[1.45] tracking-[0.045em] transition-colors',
        active
          ? 'border-signal bg-signal-deep/80 text-signal-bright'
          : 'border-ink-600 bg-white/65 text-ink-100 hover:border-ink-500 hover:text-ink-50',
      )}
    >
      {icon}
      <span className="hidden sm:inline">{label}</span>
    </button>
  );
}

/** Full-width action button. Red outline = primary investigative action. */
export function ActionButton({
  children,
  onClick,
  primary = false,
}: {
  children: ReactNode;
  onClick: () => void;
  primary?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={clsx(
        'w-full rounded-md border px-2 py-[7px] text-[11px] font-semibold uppercase leading-[1.45] tracking-[0.05em] transition-colors',
        primary
          ? 'border-signal bg-signal-deep/70 text-signal-bright hover:bg-signal-deep'
          : 'border-ink-600 bg-white/65 text-ink-100 hover:border-ink-400 hover:text-ink-50',
      )}
    >
      {children}
    </button>
  );
}

/** Centered notice on the workspace canvas. */
export function WorkspaceNotice({
  title,
  detail,
  children,
  red = false,
}: {
  title: string;
  detail?: string;
  children?: ReactNode;
  red?: boolean;
}) {
  return (
    <div className="absolute inset-0 flex items-center justify-center p-6">
      <div className="floating-panel-strong corner-ticks max-w-md px-5 py-4">
        <div className={clsx('ui-heading', red && 'text-signal-bright')}>{title}</div>
        {detail && <p className="body-text mt-1.5">{detail}</p>}
        {children && <div className="mt-3">{children}</div>}
      </div>
    </div>
  );
}
