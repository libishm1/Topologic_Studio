import FireEgress

-- Every theorem must rest on Lean's standard axioms only (propext,
-- Classical.choice, Quot.sound). `sorryAx` here would mean an unproven step.
#print axioms FireEgress.Temperature.classic_overshoots
#print axioms FireEgress.Temperature.step_between
#print axioms FireEgress.Temperature.capped_coefficient_unit
#print axioms FireEgress.Temperature.cap_inactive_from_degree_two
#print axioms FireEgress.Temperature.step_bounded
#print axioms FireEgress.Shaping.shaped_eq
#print axioms FireEgress.Shaping.shaping_preserves_order
#print axioms FireEgress.Shaping.nonzero_terminal_flips_order
#print axioms FireEgress.Heuristic.consistent
#print axioms FireEgress.Heuristic.admissible
#print axioms FireEgress.Heuristic.hazard_weight_ge_length
