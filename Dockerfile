FROM python:3.9-slim AS develop

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libstdc++6 \
    libc6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock /app/

COPY code/ /app/code

RUN uv pip install --system --no-cache .

ENV PYTHONPATH=/app/code

CMD ["python", "-m", "uvicorn", "main:app", "--host=0.0.0.0", "--port=3000", "--reload"]


FROM python:3.9-slim AS staging

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libstdc++6 \
    libc6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock /app/

COPY code/ /app/code

RUN uv pip install --system --no-cache .

ENV PYTHONPATH=/app/code

CMD ["python", "-m", "uvicorn", "main:app", "--host=0.0.0.0", "--port=3000", "--reload"]


FROM python:3.9-slim AS prod

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libstdc++6 \
    libc6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock /app/

COPY code/ /app/code

RUN uv pip install --system --no-cache .

ENV PYTHONPATH=/app/code

CMD ["python", "-m", "uvicorn", "main:app", "--host=0.0.0.0", "--port=3000"]