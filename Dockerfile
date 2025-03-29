ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE} AS builder

WORKDIR /build
COPY . /build/

# Install PDM
RUN pip install --no-cache-dir pdm

# Build wheel
RUN pdm build

# Application Layer
FROM python:3.12-slim

WORKDIR /app
COPY --from=builder /build/dist/*.whl /tmp/
# Install into site-packages from wheel
RUN pip install --no-cache-dir /tmp/*.whl

ENV PORT=80
EXPOSE 80/tcp

CMD ["sh", "-c", "uvicorn kuhl_haus.bedrock.api.app:run --host 0.0.0.0 --port ${PORT}"]
