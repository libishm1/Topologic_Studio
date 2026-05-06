# HITL PDF Summaries

## `Human.pdf`

Metadata:

- Pages: 5
- Extractable text: 15,437 characters
- Title in text: "Human-in-the-Loop AI for Strategic Decision-Making in Critical Infrastructure Systems"
- Author in text: Micheal James
- Date in text: 2026

Relevant ideas for this project:

- AI should be treated as decision support in critical infrastructure workflows, not as the final authority.
- Humans must remain responsible for interpreting, validating, and overriding algorithmic outputs.
- HITL differs from human-on-the-loop monitoring because humans participate at predefined decision points.
- Strategic or high-impact decisions need accountability, transparency, and contextual judgment.
- Risks include automation bias, cognitive overload, weak institutional capacity, and poor human-AI interface design.

Adopted rules:

- Human review is required for graph/path/fire interpretation when visual or research validity is uncertain.
- Agents must summarize evidence and ask for narrow decisions to avoid cognitive overload.
- Known bugs and residual risks must be documented after each test/debug cycle.

## `1-s2.0-S1093968726030215-main.pdf`

Metadata:

- Pages: 25
- Extractable text: 87,795 characters
- Title: "AI-driven human-in-the-loop structural optimization of gridshells"
- Authors: J. Melchiorre, A. Manuello Bertetto, G. C. Marano, S. Adriaenssens
- DOI: `10.1016/j.cacaie.2026.100044`
- Status: Elsevier journal pre-proof

Relevant ideas for this project:

- The paper combines parametric generation, optimization, machine learning, clustering, and human ratings.
- It reduces human workload by asking users to evaluate representative candidates rather than every candidate.
- Human preference is treated as an objective alongside quantitative objectives.
- Clustering and dimensionality reduction help select a small set of representative cases.
- HITL speed matters because slow loops increase fatigue and reduce engagement.

Adopted rules:

- Agents run fast autonomous browser checks first.
- Agents group evidence into a small number of categories before asking for human review.
- Humans review representative screenshots and summaries, not raw exhaustive logs.
- Agent reruns focused tests after a fix and updates bug/risk documentation.

