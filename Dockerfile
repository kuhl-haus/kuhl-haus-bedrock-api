ARG BASE_IMAGE=ghcr.io/kuhl-haus/kuhl-haus-bedrock-app:latest
FROM ${BASE_IMAGE}

WORKDIR /app
COPY . /app/
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
RUN pip install --no-cache-dir .
RUN pytest /app/tests -v


ENV PORT=80
EXPOSE 80/tcp

CMD ["sh", "-c", "uvicorn kuhl_haus.bedrock.api.app:run --host 0.0.0.0 --port ${PORT}"]
