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
