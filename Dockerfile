# Runs the agent (or the evals) without a local Python setup.
#   docker build -t agent-evals .
#   docker run -it --env-file .env agent-evals "Find the ReAct paper and save a note about it"
#   docker run --env-file .env -v ./.cache:/app/.cache --entrypoint python agent-evals -m agent_evals.evals --runs 1
FROM python:3.13-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# Dependencies first, so editing the code doesn't reinstall them on every build.
COPY pyproject.toml README.md ./
RUN mkdir -p src/agent_evals && touch src/agent_evals/__init__.py && pip install -e .

# Editable install: the code finds evals/ and .cache/ relative to the repo root (/app).
COPY src ./src
COPY evals ./evals

# Don't run as root.
RUN useradd --create-home agent && mkdir .cache && chown agent .cache
USER agent

ENTRYPOINT ["python", "-m", "agent_evals.agent.loop"]
