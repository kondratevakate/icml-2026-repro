# run03 control harness — isolated, reproducible CPU environment.
# Purpose: pin the Python/dependency environment so every candidate run
# (arm 1 = Hermes-leaf, and later arms 2-4) executes under IDENTICAL conditions.
# This isolates the ENVIRONMENT, not the agent. The agent (Hermes-leaf) still
# drives via its own toolchain; we do NOT strip execute_code/patch — those are
# legitimate reproduction tooling, not contamination.
FROM python:3.12-slim

# Non-interactive, minimal
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# System deps: git (clone official code), build tools for numpy/scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Controlled dependency set — matches TASK.md (stdlib + numpy/scipy/sympy only).
# NO scikit-learn, NO torch, NO GPU. pip install is restricted to these at runtime
# via TASK.md hard rules; the image pins versions for reproducibility.
COPY requirements.txt /work/requirements.txt
RUN python -m venv .venv && \
    .venv/bin/pip install --no-cache-dir -r requirements.txt

# Router endpoint (free hy3:free). Override at run: -e OPENAI_BASE_URL=...
# The proxy key is injected at runtime from the host, never baked in.
ENV OPENAI_BASE_URL=http://host.docker.internal:8319/v1

# Entry: a run directory is mounted at /work/run/<orid>
VOLUME ["/work/run"]
WORKDIR /work/run

# Default: print environment sanity, then hand off to the agent launcher.
CMD ["/bin/bash", "-c", ".venv/bin/python -c \"import numpy,scipy,sympy; print('env ok', numpy.__version__, scipy.__version__, sympy.__version__)\""]
