# Claude Code Testing Instructions

Before browser testing, read:

- [testing/agent-instructions-codex-claude.md](testing/agent-instructions-codex-claude.md)
- [testing/headless-browser-console-testing.md](testing/headless-browser-console-testing.md)
- [testing/browser-test-matrix.md](testing/browser-test-matrix.md)

Claude Code-specific rule: use a narrow task loop. Collect browser evidence first, classify findings, update `wiki/issues/`, then ask the human only for visual or research-validity decisions that cannot be determined from logs.

