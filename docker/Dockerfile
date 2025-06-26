FROM mcr.microsoft.com/playwright/python:v1.52.0-noble as builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

USER root

WORKDIR /app

SHELL ["/bin/bash", "-c"]

ENV UV_LINK_MODE=copy \
    DISPLAY=:99

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=README.md,target=README.md \
    --mount=type=bind,source=src,target=src \
    uv sync --frozen --no-cache --no-dev

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    xvfb \
    tini \
    wget \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Install Camoufox GeoIP
RUN uv run camoufox fetch

ADD . /app

ENTRYPOINT ["/usr/bin/tini", "--"]
