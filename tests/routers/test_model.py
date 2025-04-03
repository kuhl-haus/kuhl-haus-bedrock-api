import pytest
from unittest.mock import patch, MagicMock
from fastapi import HTTPException
from fastapi.testclient import TestClient


# Use patch to mock the module import
@pytest.fixture
def models_router():
    """Import the router after mocking AWS services"""
    with patch("kuhl_haus.bedrock.api.routers.model.api_key_auth") as patched_api_key_auth:
        api_auth = MagicMock()
        patched_api_key_auth.return_value = api_auth
        with patch('kuhl_haus.bedrock.api.routers.model.BedrockModel') as mock_model_class:
            # Set up the mock model instance
            mock_model = MagicMock()
            mock_model.list_models.return_value = [
                "anthropic.claude-3-sonnet-20240229-v1:0",
                "anthropic.claude-3-haiku-20240307-v1:0",
                "amazon.titan-text-express-v1"
            ]
            mock_model_class.return_value = mock_model

            # Now import the module
            from kuhl_haus.bedrock.api.routers import model

            # Replace the router's model with our mock
            original_model = model.chat_model
            model.chat_model = mock_model

            yield model

            # Restore original
            model.chat_model = original_model


@pytest.mark.asyncio
async def test_list_models(models_router):
    """Test listing available models."""
    # Act
    result = await models_router.list_models()

    # Assert
    models_router.chat_model.list_models.assert_called_once()
    assert len(result.data) == 3
    assert result.data[0].id == "anthropic.claude-3-sonnet-20240229-v1:0"
    assert result.data[1].id == "anthropic.claude-3-haiku-20240307-v1:0"
    assert result.data[2].id == "amazon.titan-text-express-v1"


@pytest.mark.asyncio
async def test_get_model_valid(models_router):
    """Test getting a valid model by ID."""
    # Arrange
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

    # Act
    result = await models_router.get_model(model_id)

    # Assert
    models_router.chat_model.list_models.assert_called_once()
    assert result.id == model_id


@pytest.mark.asyncio
async def test_get_model_invalid(models_router):
    """Test getting an invalid model by ID raises an exception."""
    # Arrange
    model_id = "non-existent-model"

    # Act & Assert
    with pytest.raises(HTTPException) as excinfo:
        await models_router.get_model(model_id)

    # Assert
    models_router.chat_model.list_models.assert_called_once()
    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Unsupported Model Id"


@pytest.mark.asyncio
async def test_validate_model_id_valid(models_router):
    """Test validation of a valid model ID."""
    # Arrange
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

    # Act
    # No exception should be raised
    await models_router.validate_model_id(model_id)

    # Assert
    models_router.chat_model.list_models.assert_called_once()


@pytest.mark.asyncio
async def test_validate_model_id_invalid(models_router):
    """Test validation of an invalid model ID."""
    # Arrange
    model_id = "non-existent-model"

    # Act & Assert
    with pytest.raises(HTTPException) as excinfo:
        await models_router.validate_model_id(model_id)

    # Assert
    models_router.chat_model.list_models.assert_called_once()
    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Unsupported Model Id"


def test_router_endpoints(models_router):
    """Test that the router has the expected endpoints."""
    routes = [route.path for route in models_router.router.routes]
    assert "/models" in routes  # List models endpoint
    assert "/models/{model_id}" in routes  # Get model endpoint
