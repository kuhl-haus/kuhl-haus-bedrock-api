#!python
from __future__ import annotations

import argparse
import logging
import os
import traceback

import sys
import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from kuhl_haus.metrics.env import (
    CARBON_CONFIG,
    LOG_LEVEL,
    METRIC_NAMESPACE,
    NAMESPACE_ROOT,
    THREAD_POOL_SIZE,
    POD_NAME,
)
from kuhl_haus.bedrock.app.env import (
    DEFAULT_MODEL,
    ENABLE_CROSS_REGION_INFERENCE,
    SECRET_ARN_PARAMETER,
)
from kuhl_haus.metrics.middleware.request_logging import request_metrics
from kuhl_haus.metrics.recorders.graphite_logger import GraphiteLogger, GraphiteLoggerOptions

from kuhl_haus.bedrock.api import __version__
from kuhl_haus.bedrock.api.env import (
    API_ROUTE_PREFIX,
    CORS_ALLOWED_ORIGINS,
    DESCRIPTION,
    SERVER_IP,
    SERVER_PORT,
    SUMMARY,
    TITLE,
)
from kuhl_haus.bedrock.api.routers import model, chat, embeddings


config = {
    "title": TITLE,
    "description": DESCRIPTION,
    "summary": SUMMARY,
    "version": __version__,
}

graphite_logger = GraphiteLogger(GraphiteLoggerOptions(
    application_name='bedrock_api',
    log_level=LOG_LEVEL,
    carbon_config=CARBON_CONFIG,
    thread_pool_size=THREAD_POOL_SIZE,
    namespace_root=NAMESPACE_ROOT,
    metric_namespace=METRIC_NAMESPACE,
    pod_name=POD_NAME,
))
LOGGER = graphite_logger.logger

app = FastAPI(**config)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[CORS_ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(model.router, prefix=API_ROUTE_PREFIX)
app.include_router(chat.router, prefix=API_ROUTE_PREFIX)
app.include_router(embeddings.router, prefix=API_ROUTE_PREFIX)


@app.middleware("http")
async def app_metrics(request: Request, call_next):
    return await request_metrics(
        request=request,
        call_next=call_next,
        recorder=graphite_logger
    )


@app.get("/health")
async def health():
    """For health check if needed"""
    return {
        "status": "OK",
        "version": __version__,
        "api_route_prefix": API_ROUTE_PREFIX,
        "cors_allowed_origins": CORS_ALLOWED_ORIGINS,
        "default_model": DEFAULT_MODEL,
        "enable_cross_region_inference": ENABLE_CROSS_REGION_INFERENCE,
        "secret_arn_parameter": SECRET_ARN_PARAMETER,
        "log_level": LOG_LEVEL,
        "carbon_config": CARBON_CONFIG,
        "metric_namespace": METRIC_NAMESPACE,
        "namespace_root": NAMESPACE_ROOT,
        "pod_name": POD_NAME
    }


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=400)


def parse_args(args):
    parser = argparse.ArgumentParser(description="Bedrock REST API")
    parser.add_argument(
        "--host",
        dest="host",
        type=str,
        default=SERVER_IP,
    )
    parser.add_argument(
        "--port",
        dest="port",
        type=int,
        default=SERVER_PORT,
    )
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    LOGGER.info("Service starting up... Hello!")
    try:
        msg = ("instantiate from python version %s executable %s, pid: %s" %
               (sys.version, sys.executable, str(os.getpid()),))
        msg = msg.replace("\n", " ")
        LOGGER.info(msg)
        LOGGER.info("Listening on %s:%d", args.host, args.port)
        uvicorn.run(
            "kuhl_haus.bedrock.api.app:app",
            host=args.host,
            port=args.port,
            reload=True,
            access_log=True,
            log_level=logging.ERROR
        )
    except KeyboardInterrupt:
        LOGGER.info("Received interrupt, exiting")
    except Exception as e:
        print(f"Unhandled exception raised:{repr(e)}", file=sys.stderr)
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=2, file=sys.stderr)
        LOGGER.error("Unhandled exception raised:%s", repr(e), exc_info=e, stack_info=True)
        raise
    finally:
        LOGGER.info("Service shutting down... Good-bye.")


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[2:])


if __name__ == "__main__":
    # ^  This is a guard statement that will prevent the following code from
    #    being executed in the case someone imports this file instead of
    #    executing it as a script.
    #    https://docs.python.org/3/library/__main__.html

    # After installing your project with pip, users can also run your Python
    # modules as scripts via the ``-m`` flag, as defined in PEP 338::
    #
    #     python -m bedrock_api.skeleton 42
    #
    run()
