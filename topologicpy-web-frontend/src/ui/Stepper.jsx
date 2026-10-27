/**
 * The workflow as a tab list: Model, Graph, Route, Simulate. Each step shows
 * its number, or a tick when done and a mark when done with a warning, and
 * says so in words for screen readers. Every step stays selectable: a later
 * step explains what it is waiting for rather than being locked.
 *
 * Keyboard: the tab pattern (WAI-ARIA APG) - arrow keys, Home and End move
 * between steps; only the selected step is in the Tab order.
 */
import React, { useRef } from "react";

import { Icon } from "./primitives.jsx";

const SPOKEN = { done: "done", warning: "done, with a warning", todo: "not done yet" };

export function Stepper({ steps, active, status, onSelect }) {
  const refs = useRef({});

  const move = (event, index) => {
    let next = null;
    if (event.key === "ArrowRight" || event.key === "ArrowDown") next = (index + 1) % steps.length;
    else if (event.key === "ArrowLeft" || event.key === "ArrowUp") next = (index - 1 + steps.length) % steps.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = steps.length - 1;
    if (next === null) return;
    event.preventDefault();
    onSelect(steps[next].id);
    refs.current[steps[next].id]?.focus();
  };

  return (
    <div className="stepper" role="tablist" aria-label="Workflow steps">
      {steps.map((step, index) => {
        const state = status[step.id] || "todo";
        const selected = active === step.id;
        return (
          <button
            key={step.id}
            ref={(el) => {
              refs.current[step.id] = el;
            }}
            id={`step-${step.id}`}
            type="button"
            role="tab"
            className="stepper__step"
            aria-selected={selected}
            aria-controls="workflow-panel"
            tabIndex={selected ? 0 : -1}
            data-step={step.id}
            data-status={state}
            onClick={() => onSelect(step.id)}
            onKeyDown={(event) => move(event, index)}
          >
            <span className="stepper__marker" aria-hidden="true">
              {state === "done" ? (
                <Icon name="check" size={14} />
              ) : state === "warning" ? (
                <Icon name="alert" size={14} />
              ) : (
                index + 1
              )}
            </span>
            <span>{step.label}</span>
            <span className="visually-hidden">
              , step {index + 1} of {steps.length}, {SPOKEN[state]}
            </span>
          </button>
        );
      })}
    </div>
  );
}
