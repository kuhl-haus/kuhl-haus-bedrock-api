import os
import pytest
from unittest.mock import patch

import kuhl_haus.bedrock.api.env as env


@pytest.fixture
def reset_env_vars():
    """Fixture to reset environment variables after test."""
    original_values = {}
    env_vars = [
        "API_ROUTE_PREFIX",
        "SERVER_IP",
        "SERVER_PORT",
        "CORS_ALLOWED_ORIGINS"
    ]

    # Save original values
    for var in env_vars:
        original_values[var] = os.environ.get(var)

    yield

    # Restore original values
    for var, value in original_values.items():
        if value is None:
            if var in os.environ:
                del os.environ[var]
        else:
            os.environ[var] = value


def test_default_constants_without_env_vars(reset_env_vars):
    """Test that constants have expected default values when env vars aren't set."""
    # Arrange
    env_vars = [
        "API_ROUTE_PREFIX",
        "SERVER_IP",
        "SERVER_PORT",
        "CORS_ALLOWED_ORIGINS"
    ]

    for var in env_vars:
        if var in os.environ:
            del os.environ[var]

    # Act - Import the module again to re-evaluate constants
    import importlib
    importlib.reload(env)

    # Assert
    assert env.API_ROUTE_PREFIX == "/api/v1"
    assert env.SERVER_IP == "0.0.0.0"
    assert env.SERVER_PORT == 80
    assert env.CORS_ALLOWED_ORIGINS == "*"


@patch.dict(os.environ, {
    "API_ROUTE_PREFIX": "/custom/api",
    "SERVER_IP": "127.0.0.1",
    "SERVER_PORT": "8080",
    "CORS_ALLOWED_ORIGINS": "http://localhost:3000"
})
def test_constants_with_custom_env_vars():
    """Test that constants use values from environment variables when set."""
    # Arrange & Act - Import the module again to re-evaluate constants
    import importlib
    importlib.reload(env)

    # Assert
    assert env.API_ROUTE_PREFIX == "/custom/api"
    assert env.SERVER_IP == "127.0.0.1"
    assert env.SERVER_PORT == "8080"
    assert env.CORS_ALLOWED_ORIGINS == "http://localhost:3000"


def test_documentation_constants():
    """Test that documentation constants are set correctly."""
    # Arrange - These should be static values not from environment
    sut = env

    # Act & Assert
    assert sut.TITLE == "Amazon Bedrock Proxy APIs"
    assert sut.SUMMARY == "OpenAI-Compatible RESTful APIs for Amazon Bedrock"
    assert "OpenAI-Compatible REST API" in sut.DESCRIPTION


@patch.dict(os.environ, {"SERVER_PORT": "invalid_port"})
def test_non_numeric_server_port():
    """Test that non-numeric SERVER_PORT is handled correctly."""
    # Arrange & Act - Import the module to evaluate constants
    import importlib
    importlib.reload(env)

    # Assert - The module should still load, and SERVER_PORT should be the string from the env var
    assert env.SERVER_PORT == "invalid_port"


@patch.dict(os.environ, {"API_ROUTE_PREFIX": ""})
def test_empty_string_env_var():
    """Test that empty string in environment variable is respected."""
    # Arrange & Act - Import the module to evaluate constants
    import importlib
    importlib.reload(env)

    # Assert - Empty string should be used as-is
    assert env.API_ROUTE_PREFIX == ""
