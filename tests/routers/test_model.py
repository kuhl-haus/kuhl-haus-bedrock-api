from unittest.mock import patch

import pytest
from fastapi import HTTPException
from kuhl_haus.bedrock.app.schema import Models, Model

from kuhl_haus.bedrock.api.routers.model import list_models, get_model, validate_model_id


@pytest.fixture
def mock_chat_model():
    """Fixture for a mocked BedrockModel instance."""
    with patch("kuhl_haus.bedrock.api.routers.model.chat_model") as mock_model:
        mock_model.list_models.return_value = [
            "anthropic.claude-3-sonnet-20240229-v1:0",
            "anthropic.claude-3-haiku-20240307-v1:0",
            "amazon.titan-text-express-v1"
        ]
        yield mock_model


@pytest.mark.asyncio
async def test_list_models(mock_chat_model):
    """Test listing available models."""
    # Act
    result = await list_models()

    # Assert
    mock_chat_model.list_models.assert_called_once()
    assert isinstance(result, Models)
    assert len(result.data) == 3
    assert all(isinstance(model, Model) for model in result.data)
    assert result.data[0].id == "anthropic.claude-3-sonnet-20240229-v1:0"
    assert result.data[1].id == "anthropic.claude-3-haiku-20240307-v1:0"
    assert result.data[2].id == "amazon.titan-text-express-v1"


@pytest.mark.asyncio
async def test_get_model_valid(mock_chat_model):
    """Test getting a valid model by ID."""
    # Arrange
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

    # Act
    result = await get_model(model_id)

    # Assert
    mock_chat_model.list_models.assert_called_once()
    assert isinstance(result, Model)
    assert result.id == model_id


@pytest.mark.asyncio
async def test_get_model_invalid(mock_chat_model):
    """Test getting an invalid model by ID raises an exception."""
    # Arrange
    model_id = "non-existent-model"

    # Act & Assert
    with pytest.raises(HTTPException) as excinfo:
        await get_model(model_id)

    # Assert
    mock_chat_model.list_models.assert_called_once()
    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Unsupported Model Id"


@pytest.mark.asyncio
async def test_validate_model_id_valid(mock_chat_model):
    """Test validation of a valid model ID."""
    # Arrange
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

    # Act
    # No exception should be raised
    await validate_model_id(model_id)

    # Assert
    mock_chat_model.list_models.assert_called_once()


@pytest.mark.asyncio
async def test_validate_model_id_invalid(mock_chat_model):
    """Test validation of an invalid model ID."""
    # Arrange
    model_id = "non-existent-model"

    # Act & Assert
    with pytest.raises(HTTPException) as excinfo:
        await validate_model_id(model_id)

    # Assert
    mock_chat_model.list_models.assert_called_once()
    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Unsupported Model Id"


@pytest.mark.asyncio
async def test_list_models_empty(mock_chat_model):
    """Test listing models when none are available."""
    # Arrange
    mock_chat_model.list_models.return_value = []

    # Act
    result = await list_models()

    # Assert
    mock_chat_model.list_models.assert_called_once()
    assert isinstance(result, Models)
    assert len(result.data) == 0


@pytest.mark.asyncio
async def test_list_models_single(mock_chat_model):
    """Test listing models when only one is available."""
    # Arrange
    mock_chat_model.list_models.return_value = ["anthropic.claude-3-sonnet-20240229-v1:0"]

    # Act
    result = await list_models()

    # Assert
    mock_chat_model.list_models.assert_called_once()
    assert isinstance(result, Models)
    assert len(result.data) == 1
    assert result.data[0].id == "anthropic.claude-3-sonnet-20240229-v1:0"
