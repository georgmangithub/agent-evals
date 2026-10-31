# agent-evals

A tool-using AI agent built like production software: its own MCP server, an eval harness with regression tests in CI, tracing with cost and latency per task, and guardrails against prompt injection.

Goal: show the engineering around the model, not just a chatbot.

## Architecture (planned)

```
user task -> agent loop -> tools (own MCP server) -> result
                |               |
            tracing         guardrails (permissions, injection checks, human approval)
                |
          eval harness (task suite, judge, metrics, CI)
```

## Roadmap

- [x] Model client: Gemini free tier or Ollama, with every response cached on disk
- [x] Own MCP server: arXiv search, paper lookup, notes in SQLite
- [ ] Agent loop that calls the MCP tools
- [ ] Tracing: per-step traces, cost and latency per task
- [ ] Guardrails: human approval for destructive tools, a few prompt-injection tests
- [ ] Small eval suite: ~5 tasks with known answers and a script that prints a score
- [ ] CI on GitHub Actions running the tests and eval suite
- [ ] Docker + deployed demo (Hugging Face Spaces / Streamlit Cloud)
- [ ] Write-up: tradeoffs, failures, fixes

## Cost notes
- Develop on free tiers or local models; cache every API response.
- Run the final benchmark once on a paid frontier model; set a hard spending cap.
