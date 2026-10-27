/**
 * Small presentational building blocks.
 *
 * Everything the panels need, so no panel reaches for a raw styled div and
 * the spacing/typography stays consistent across the app.
 */
import React, { useId, useState } from "react";

export function Button({
  children,
  variant = "default",
  size = "md",
  block = false,
  busy = false,
  icon = false,
  className = "",
  ...rest
}) {
  const classes = [
    "btn",
    variant !== "default" && `btn--${variant}`,
    size !== "md" && `btn--${size}`,
    block && "btn--block",
    icon && "btn--icon",
    className,
  ]
    .filter(Boolean)
    .join(" ");
  return (
    <button type="button" className={classes} disabled={busy || rest.disabled} {...rest}>
      {busy && <span className="btn__spinner" aria-hidden="true" />}
      {children}
    </button>
  );
}

export function FileButton({ accept, onFile, children, variant = "default", disabled }) {
  return (
    <label
      className={`btn file-btn${variant !== "default" ? ` btn--${variant}` : ""}${
        disabled ? " btn--disabled" : ""
      }`}
      style={disabled ? { opacity: 0.45, pointerEvents: "none" } : undefined}
    >
      <input
        type="file"
        accept={accept}
        disabled={disabled}
        onChange={(event) => {
          const file = event.target.files?.[0];
          // Reset so choosing the same file twice still fires a change event.
          event.target.value = "";
          if (file) onFile(file);
        }}
      />
      <span>{children}</span>
    </label>
  );
}

export function Segmented({ value, onChange, options, label }) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          className="seg__item"
          aria-pressed={value === option.value}
          onClick={() => onChange(option.value)}
          title={option.hint}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

export function Section({ title, badge, children, defaultOpen = true, id }) {
  const [open, setOpen] = useState(defaultOpen);
  const bodyId = useId();
  return (
    <section className="section" id={id}>
      <button
        type="button"
        className="section__header"
        aria-expanded={open}
        aria-controls={bodyId}
        onClick={() => setOpen((v) => !v)}
      >
        <svg
          className="section__chevron"
          width="10"
          height="10"
          viewBox="0 0 10 10"
          aria-hidden="true"
        >
          <path d="M3 1l4 4-4 4" fill="none" stroke="currentColor" strokeWidth="1.6" />
        </svg>
        <span>{title}</span>
        {badge !== undefined && badge !== null && (
          <span className="section__badge">{badge}</span>
        )}
      </button>
      {open && (
        <div className="section__body" id={bodyId}>
          {children}
        </div>
      )}
    </section>
  );
}

export function Slider({ label, value, min, max, step, onChange, format, hint, disabled }) {
  const display = format ? format(value) : value;
  return (
    <div className="field">
      <div className="field__label">
        <span>{label}</span>
        <span className="field__value">{display}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.target.value))}
        aria-label={label}
      />
      {hint && <p className="field__hint">{hint}</p>}
    </div>
  );
}

export function NumberField({ label, value, onChange, hint, ...rest }) {
  return (
    <div className="field">
      <div className="field__label">
        <span>{label}</span>
      </div>
      <input
        type="number"
        value={value ?? ""}
        onChange={(event) =>
          onChange(event.target.value === "" ? null : Number(event.target.value))
        }
        aria-label={label}
        {...rest}
      />
      {hint && <p className="field__hint">{hint}</p>}
    </div>
  );
}

export function SelectField({ label, value, onChange, options, hint }) {
  return (
    <div className="field">
      <div className="field__label">
        <span>{label}</span>
      </div>
      <select value={value} onChange={(event) => onChange(event.target.value)} aria-label={label}>
        {options.map((option) => (
          <option key={option.value} value={option.value} disabled={option.disabled}>
            {option.label}
          </option>
        ))}
      </select>
      {hint && <p className="field__hint">{hint}</p>}
    </div>
  );
}

export function Toggle({ label, checked, onChange, hint, disabled }) {
  return (
    <div className="field">
      <label className="toggle">
        <input
          type="checkbox"
          checked={checked}
          disabled={disabled}
          onChange={(event) => onChange(event.target.checked)}
        />
        <span className="toggle__track" aria-hidden="true" />
        <span className="toggle__label">{label}</span>
      </label>
      {hint && <p className="field__hint">{hint}</p>}
    </div>
  );
}

export function Stat({ label, value, warn = false }) {
  return (
    <div className="stat">
      <div className="stat__label">{label}</div>
      <div className={`stat__value${warn ? " stat__value--warn" : ""}`}>{value}</div>
    </div>
  );
}

export function Badge({ children, variant, pulse = false }) {
  return (
    <span className={`badge${variant ? ` badge--${variant}` : ""}`}>
      {variant && (
        <span className={`badge__dot${pulse ? " badge__dot--pulse" : ""}`} aria-hidden="true" />
      )}
      {children}
    </span>
  );
}

export function KeyValue({ entries }) {
  if (!entries?.length) return null;
  return (
    <div className="kv">
      {entries.map(([key, value]) => (
        <React.Fragment key={key}>
          <div className="kv__k">{key}</div>
          <div className="kv__v">{value}</div>
        </React.Fragment>
      ))}
    </div>
  );
}

export function Empty({ children }) {
  return <p className="empty">{children}</p>;
}

// ------------------------------------------------------------- redesign

/**
 * Stroke icons, drawn inline so they take the text colour. Decorative unless
 * a label is given; icon-only buttons carry their own aria-label instead.
 */
const ICON_PATHS = {
  check: <path d="M4 10.5 8 14.5 16 6" />,
  alert: (
    <>
      <path d="M10 3 18 17H2Z" />
      <path d="M10 8.5v4" />
      <circle cx="10" cy="14.8" r="0.5" fill="currentColor" />
    </>
  ),
  info: (
    <>
      <circle cx="10" cy="10" r="7.5" />
      <path d="M10 9v5" />
      <circle cx="10" cy="6.5" r="0.5" fill="currentColor" />
    </>
  ),
  help: (
    <>
      <circle cx="10" cy="10" r="8" />
      <path d="M7.6 7.6a2.5 2.5 0 0 1 4.8.9c0 1.7-2.4 2-2.4 3.5" />
      <circle cx="10" cy="14.6" r="0.5" fill="currentColor" />
    </>
  ),
  sun: (
    <>
      <circle cx="10" cy="10" r="3.5" />
      <path d="M10 2v2M10 16v2M2 10h2M16 10h2M4.3 4.3l1.4 1.4M14.3 14.3l1.4 1.4M4.3 15.7l1.4-1.4M14.3 5.7l1.4-1.4" />
    </>
  ),
  moon: <path d="M16 12.5A6.5 6.5 0 0 1 7.5 4a6.5 6.5 0 1 0 8.5 8.5Z" />,
  panel: (
    <>
      <rect x="2.5" y="3.5" width="15" height="13" rx="2" />
      <path d="M12.5 3.5v13" />
    </>
  ),
  upload: (
    <>
      <path d="M10 13V3M6 7l4-4 4 4" />
      <path d="M3 13v2.5A1.5 1.5 0 0 0 4.5 17h11a1.5 1.5 0 0 0 1.5-1.5V13" />
    </>
  ),
  close: <path d="M5 5l10 10M15 5 5 15" />,
  book: (
    <>
      <path d="M3 4.5A1.5 1.5 0 0 1 4.5 3H9v14H4.5A1.5 1.5 0 0 1 3 15.5Z" />
      <path d="M17 4.5A1.5 1.5 0 0 0 15.5 3H11v14h4.5a1.5 1.5 0 0 0 1.5-1.5Z" />
    </>
  ),
};

export function Icon({ name, size = 18, label, className }) {
  return (
    <svg
      className={className}
      width={size}
      height={size}
      viewBox="0 0 20 20"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      role={label ? "img" : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : "true"}
      focusable="false"
    >
      {ICON_PATHS[name]}
    </svg>
  );
}

/**
 * A message that explains a result, kept beside it rather than in a toast:
 * toasts vanish and cover the model, and these need reading.
 */
export function Notice({ tone = "neutral", children, live = false }) {
  const icon = tone === "warning" ? "alert" : tone === "success" ? "check" : "info";
  return (
    <div className={`notice notice--${tone}`} role={live ? "status" : undefined}>
      <Icon name={icon} size={18} />
      <div>{children}</div>
    </div>
  );
}

/** "Step 2 of 4" over the step's name and one line on what it is for. */
export function StepHeading({ step, of = 4, title, lead }) {
  return (
    <div className="step-heading">
      <p className="step-heading__count">
        Step {step} of {of}
      </p>
      <h2 className="step-heading__title">{title}</h2>
      {lead && <p className="step-heading__lead">{lead}</p>}
    </div>
  );
}

/** Radio buttons as cards, each with a line saying what the choice does. */
export function Choices({ legend, name, value, onChange, options, disabled }) {
  return (
    <fieldset className="choices" disabled={disabled}>
      <legend className="choices__legend">{legend}</legend>
      {options.map((option) => (
        <label className="choice" key={option.value}>
          <input
            type="radio"
            name={name}
            value={option.value}
            checked={value === option.value}
            onChange={() => onChange(option.value)}
          />
          <span>
            <span className="choice__title">{option.label}</span>
            {option.description && <span className="choice__desc">{option.description}</span>}
          </span>
        </label>
      ))}
    </fieldset>
  );
}

/** Settings that most people leave alone, folded under one line. */
export function Disclosure({ summary, children, defaultOpen = false }) {
  return (
    <details className="disclosure" open={defaultOpen || undefined}>
      <summary>{summary}</summary>
      <div className="disclosure__body">{children}</div>
    </details>
  );
}
