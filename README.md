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


## Cost notes
- Develop on free tiers or local models; cache every API response.
- Run the final benchmark once on a paid frontier model; set a hard spending cap.
