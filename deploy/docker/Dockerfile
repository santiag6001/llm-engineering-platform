# syntax=docker/dockerfile:1
ARG PYTHON_IMAGE=python:3.12.11-slim-bookworm

FROM ${PYTHON_IMAGE} AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN python -m venv /opt/venv
COPY requirements/runtime.txt requirements/runtime.txt
RUN /opt/venv/bin/python -m pip install --no-cache-dir -r requirements/runtime.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip wheel --no-deps --wheel-dir /tmp/wheels . \
    && /opt/venv/bin/python -m pip install --no-cache-dir --no-deps \
        /tmp/wheels/llm_production_platform-*.whl \
    && /opt/venv/bin/python -m pip check \
    && rm -rf /opt/venv/lib/python3.12/site-packages/pip \
        /opt/venv/lib/python3.12/site-packages/pip-*.dist-info \
        /opt/venv/bin/pip /opt/venv/bin/pip3 /opt/venv/bin/pip3.12

FROM ${PYTHON_IMAGE} AS runtime

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --no-create-home --home-dir /nonexistent app

COPY --from=builder /opt/venv /opt/venv

USER 10001:10001
WORKDIR /app

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).read()"]

CMD ["uvicorn", "llm_platform.main:app", "--host", "0.0.0.0", "--port", "8000"]
