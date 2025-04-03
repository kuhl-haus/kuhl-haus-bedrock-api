import pytest
import pytest_asyncio
from unittest.mock import Mock, patch
from fastapi import HTTPException

from kuhl_haus.bedrock.api.routers.embeddings import embeddings
from kuhl_haus.bedrock.app.schema import EmbeddingsRequest, EmbeddingsResponse
from kuhl_haus.bedrock.app.env import DEFAULT_EMBEDDING_MODEL


@pytest.fixture
def mock_embeddings_request():
    """Fixture for a valid EmbeddingsRequest object."""
    mock = Mock(spec=EmbeddingsRequest)
    mock.model = "cohere.embed-multilingual-v3"
    mock.input = ["Your text string goes here"]
    return mock


@pytest.mark.asyncio
@patch("kuhl_haus.bedrock.api.routers.embeddings.get_embeddings_model")
async def test_embeddings_successful(patched_get_embeddings_model, mock_embeddings_request):
    """Test successful embedding generation."""
    # Arrange
    mock_model = Mock()
    mock_response = Mock(spec=EmbeddingsResponse)
    mock_model.embed.return_value = mock_response
    patched_get_embeddings_model.return_value = mock_model

    # Act
    result = await embeddings(mock_embeddings_request)

    # Assert
    patched_get_embeddings_model.assert_called_once_with(mock_embeddings_request.model)
    mock_model.embed.assert_called_once_with(mock_embeddings_request)
    assert result == mock_response


@pytest.mark.asyncio
@patch("kuhl_haus.bedrock.api.routers.embeddings.get_embeddings_model")
async def test_embeddings_openai_model_replacement(patched_get_embeddings_model, mock_embeddings_request):
    """Test that OpenAI embedding models are replaced with the default model."""
    # Arrange
    mock_embeddings_request.model = "text-embedding-ada-002"
    mock_model = Mock()
    mock_response = Mock(spec=EmbeddingsResponse)
    mock_model.embed.return_value = mock_response
    patched_get_embeddings_model.return_value = mock_model

    # Act
    await embeddings(mock_embeddings_request)

    # Assert
    assert mock_embeddings_request.model == DEFAULT_EMBEDDING_MODEL
    patched_get_embeddings_model.assert_called_once_with(DEFAULT_EMBEDDING_MODEL)
    mock_model.embed.assert_called_once_with(mock_embeddings_request)


@pytest.mark.asyncio
@patch("kuhl_haus.bedrock.api.routers.embeddings.get_embeddings_model")
async def test_embeddings_unsupported_model(patched_get_embeddings_model, mock_embeddings_request):
    """Test handling of unsupported embedding models."""
    # Arrange
    model_error = ValueError("Unsupported embedding model")
    patched_get_embeddings_model.side_effect = model_error

    # Act & Assert
    with pytest.raises(ValueError) as excinfo:
        await embeddings(mock_embeddings_request)

    assert str(excinfo.value) == "Unsupported embedding model"
    patched_get_embeddings_model.assert_called_once_with(mock_embeddings_request.model)


@pytest.mark.asyncio
@patch("kuhl_haus.bedrock.api.routers.embeddings.get_embeddings_model")
async def test_embeddings_generation_error(patched_get_embeddings_model, mock_embeddings_request):
    """Test handling of errors during embedding generation."""
    # Arrange
    mock_model = Mock()
    embed_error = RuntimeError("Embedding generation failed")
    mock_model.embed.side_effect = embed_error
    patched_get_embeddings_model.return_value = mock_model

    # Act & Assert
    with pytest.raises(RuntimeError) as excinfo:
        await embeddings(mock_embeddings_request)

    assert str(excinfo.value) == "Embedding generation failed"
    patched_get_embeddings_model.assert_called_once_with(mock_embeddings_request.model)
    mock_model.embed.assert_called_once_with(mock_embeddings_request)


@pytest.mark.asyncio
@patch("kuhl_haus.bedrock.api.routers.embeddings.get_embeddings_model")
async def test_embeddings_case_insensitive_model_replacement(patched_get_embeddings_model, mock_embeddings_request):
    """Test that model replacement is case-insensitive."""
    # Arrange
    mock_embeddings_request.model = "TEXT-EMBEDDING-ada-002"
    mock_model = Mock()
    mock_response = Mock(spec=EmbeddingsResponse)
    mock_model.embed.return_value = mock_response
    patched_get_embeddings_model.return_value = mock_model

    # Act
    await embeddings(mock_embeddings_request)

    # Assert
    assert mock_embeddings_request.model == DEFAULT_EMBEDDING_MODEL
    patched_get_embeddings_model.assert_called_once_with(DEFAULT_EMBEDDING_MODEL)
    mock_model.embed.assert_called_once_with(mock_embeddings_request)


@pytest.mark.asyncio
@patch("kuhl_haus.bedrock.api.routers.embeddings.get_embeddings_model")
async def test_embeddings_with_multiple_inputs(patched_get_embeddings_model, mock_embeddings_request):
    """Test embedding generation with multiple input strings."""
    # Arrange
    mock_embeddings_request.input = ["First text", "Second text", "Third text"]
    mock_model = Mock()
    mock_response = Mock(spec=EmbeddingsResponse)
    mock_model.embed.return_value = mock_response
    patched_get_embeddings_model.return_value = mock_model

    # Act
    result = await embeddings(mock_embeddings_request)

    # Assert
    patched_get_embeddings_model.assert_called_once_with(mock_embeddings_request.model)
    mock_model.embed.assert_called_once_with(mock_embeddings_request)
    assert result == mock_response
