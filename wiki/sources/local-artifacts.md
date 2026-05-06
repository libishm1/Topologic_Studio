# Local Artifacts

## PDF

File:

```text
the synergy of non manifold topology.pdf
```

Parse status:

- Size: `2,864,899` bytes.
- Pages: `10`.
- Extractable text characters: `0`.
- Embedded images reported by PyMuPDF: `113`.
- Metadata title: `ecaadesigradi2019_671.pdf`.
- Metadata author: `Libish Murugesan`.
- Created: `2026-01-05 13:50:54 +03:00`.

Conclusion: this PDF is image-based or otherwise has no extractable text in the local copy. OCR is required before semantic summarization.

## HITL PDFs

Files:

```text
C:\Users\lmurugesan\OneDrive - Alfaisal University\CM-iTAD\topologic_webapp\Human.pdf
C:\Users\lmurugesan\OneDrive - Alfaisal University\CM-iTAD\topologic_webapp\1-s2.0-S1093968726030215-main.pdf
```

Parse status:

- `Human.pdf`: 5 pages, 15,437 extractable text characters, no embedded images reported, metadata creator `Microsoft Word 2010`.
- `1-s2.0-S1093968726030215-main.pdf`: 25 pages, 87,795 extractable text characters, 148 embedded images, Elsevier pre-proof metadata for "AI-driven human-in-the-loop structural optimization of gridshells", DOI `10.1016/j.cacaie.2026.100044`.

Documentation derived from these PDFs:

- [HITL testing principles](../testing/human-in-the-loop-principles.md)
- [Agent instructions for Codex and Claude Code](../testing/agent-instructions-codex-claude.md)
- [Headless browser console testing](../testing/headless-browser-console-testing.md)

## Logs

- `frontend-dev.log`: Vite dev server startup output.
- `frontend-dev.err.log`: present and empty.
- `uvicorn-out.log`: Uvicorn health and an old failed topology upload.
- `uvicorn-err.log`: old stack trace for `/upload-topology`.

## Sample Data

- `sample_topology_contract.json`: large viewer contract sample.
- `Code01.ipynb`: notebook artifact, not parsed for this wiki.

## Generated Verification Build

The latest frontend production build was written to:

```text
wiki/verification/frontend-dist/
```

Do not load this directory as normal documentation context.
