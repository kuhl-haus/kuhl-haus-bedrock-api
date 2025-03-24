ARG BASE_IMAGE=python:3.12
FROM ${BASE_IMAGE} AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir pdm

WORKDIR /build
COPY . /build/

# Install dependencies and build/install package
RUN pdm install -G testing

# Run tests
RUN pdm run pytest tests -v

# Build wheel
RUN pdm build

# Application Layer
FROM python:3.12

WORKDIR /app
COPY --from=builder /build/dist/*.whl /tmp/
# Install into site-packages from wheel
RUN pip install --no-cache-dir /tmp/*.whl

# ARG BASE_IMAGE=ghcr.io/kuhl-haus/kuhl-haus-bedrock-app:latest
# FROM ${BASE_IMAGE}
#
# WORKDIR /app
# COPY . /app/
# RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
# RUN pip install --no-cache-dir .
# RUN pytest /app/tests -v


ENV PORT=80
EXPOSE 80/tcp

CMD ["sh", "-c", "uvicorn kuhl_haus.bedrock.api.app:run --host 0.0.0.0 --port ${PORT}"]
