import os


TITLE = "Amazon Bedrock Proxy APIs"
SUMMARY = "OpenAI-Compatible RESTful APIs for Amazon Bedrock"
DESCRIPTION = """
This service provides an OpenAI-Compatible REST API utilizing Amazon Bedrock for the back-end.
"""

# API Settings
API_ROUTE_PREFIX = os.environ.get("API_ROUTE_PREFIX", "/api/v1")
SERVER_IP = os.environ.get("SERVER_IP", "0.0.0.0")
SERVER_PORT = os.environ.get("SERVER_PORT", 80)
CORS_ALLOWED_ORIGINS = os.environ.get("CORS_ALLOWED_ORIGINS", "*")
