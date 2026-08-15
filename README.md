# cloud-agent-team ☁️🤖

An **autonomous, multi-model AI software-engineering team** that runs entirely inside GitHub Actions. On a schedule it plans, writes, and debugs code hands-off, committing its progress back to the repository.

> Sister project to [agent-team](https://github.com/vanshulgoyal101/agent-team) — this variant is provider-agnostic (Gemini / OpenAI / Groq) and cloud-native.

---

## How it works

A small crew of specialised agents collaborate through **structured (Pydantic-typed) messages**:

| Agent | Role |
|-------|------|
| 🧭 **Architect** | Breaks the goal into a `Backlog` of concrete, file-scoped tasks. |
| 👩‍💻 **Coder** | Implements each task, emitting full file contents as `FileAction`s. |
| 🐞 **Debugger** | Runs the workspace, reads failures, and proposes fixes. |

State is persisted between runs in `workspace/state.json`, and generated code lands under `workspace/src` and `workspace/tests`.

## Automation

`workflow.yml` triggers the team on a **30-minute cron** (and via manual `workflow_dispatch`), installs dependencies, runs one iteration, and commits the result:

```yaml
on:
  schedule:
    - cron: '*/30 * * * *'
  workflow_dispatch:
```

## Configuration

API keys are read from the environment (and never committed):

```bash
export GEMINI_API_KEY=...   # or OPENAI_API_KEY / GROQ_API_KEY
python agent_team.py
```

> ⚠️ **Security:** `keys.json` is git-ignored on purpose. Never commit provider keys — use environment variables or GitHub Actions secrets, and rotate any key that has been exposed.

## Local run

```bash
pip install -r requirements.txt
python agent_team.py
```

## License

[MIT](./LICENSE) © Vanshul Goyal
