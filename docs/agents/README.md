# AI agents

Optional, tool-agnostic AI agents for working on ChaBo-Orchestrator. Nothing runs
automatically: a developer invokes an agent from their own AI coding tool.

| Agent | What it does | Permissions |
|---|---|---|
| [change-reviewer](change-reviewer.md) | Reviews a change for bugs, missing tests, secrets and docs/repo follow-ups | read-only |
| [security-auditor](security-auditor.md) | Audits the repo for secrets, internal info, unsafe routes and dependency risks | read-only |
| [test-runner](test-runner.md) | Runs local validation checks in a credential-free, disposable environment | runs commands, no edits |

## Layout

The instructions live **once**, in this folder. Each AI tool gets a thin stub that
declares the agent (name, description, allowed tools) and points back here.

```
docs/agents/<agent>.md               single source of the instructions
.github/agents/<agent>.agent.md      GitHub Copilot stub
.claude/agents/<agent>.md            Claude Code stub
```

## Changing or adding an agent

- **Change behaviour:** edit `docs/agents/<agent>.md` only. Never put instructions in a stub.
- **Add an agent:** add `docs/agents/<agent>.md` plus one stub per supported tool.
- **Support another tool:** add a stub folder in that tool's format. Keep each stub's
  allowed tools equivalent to the existing ones (read-only agents must stay read-only).

| Permission | Copilot | Claude Code |
|---|---|---|
| Read files | `read` | `Read` |
| Search | `search` | `Grep`, `Glob` |
| Run commands | `execute` | `Bash` |
