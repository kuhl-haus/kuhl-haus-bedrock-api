from unittest.mock import Mock, MagicMock, patch

import pytest
from fastapi import Depends
from fastapi.responses import StreamingResponse
from kuhl_haus.bedrock.app.env import DEFAULT_MODEL
from kuhl_haus.bedrock.app.schema import ChatRequest, ChatResponse

from kuhl_haus.bedrock.api.routers.chat import chat_completions


@pytest.fixture
def mock_chat_request():
    """Fixture for a valid ChatRequest object."""
    mock = Mock(spec=ChatRequest)
    mock.model = "anthropic.claude-3-sonnet-20240229-v1:0"
    mock.messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    mock.stream = False
    return mock


@pytest.mark.asyncio
@patch("kuhl_haus.bedrock.api.routers.chat.BedrockModel")
@patch("kuhl_haus.bedrock.api.routers.chat.api_key_auth")
@patch("kuhl_haus.bedrock.api.routers.chat.Depends")
async def test_chat_completions_non_streaming(fastapi_dep, patched_api_key_auth, patched_bedrock_model, mock_chat_request):
    """Test the chat_completions endpoint with non-streaming request."""
    # Arrange
    mock_dep = MagicMock(spec=Depends)
    fastapi_dep.return_value = mock_dep
    api_auth = MagicMock()
    patched_api_key_auth.return_value = api_auth
    mock_model = Mock()
    mock_response = Mock(spec=ChatResponse)
    mock_model.chat.return_value = mock_response
    mock_model.validate.return_value = None
    patched_bedrock_model.return_value = mock_model

    # Act
    result = await chat_completions(mock_chat_request)

    # Assert
    patched_bedrock_model.assert_called_once()
    mock_model.validate.assert_called_once_with(mock_chat_request)
    mock_model.chat.assert_called_once_with(mock_chat_request)
    assert result == mock_response
    assert not isinstance(result, StreamingResponse)


@pytest.mark.asyncio
@patch("kuhl_haus.bedrock.api.routers.chat.BedrockModel")
@patch("kuhl_haus.bedrock.api.routers.chat.api_key_auth")
@patch("kuhl_haus.bedrock.api.routers.chat.Depends")
async def test_chat_completions_streaming(fastapi_dep, patched_api_key_auth, patched_bedrock_model, mock_chat_request):
    """Test the chat_completions endpoint with streaming request."""
    # Arrange
    mock_dep = MagicMock(spec=Depends)
    fastapi_dep.return_value = mock_dep
    api_auth = MagicMock()
    patched_api_key_auth.return_value = api_auth
    mock_chat_request.stream = True
    mock_model = Mock()
    mock_stream_generator = (chunk for chunk in ["Hello", " there", "!"])
    mock_model.chat_stream.return_value = mock_stream_generator
    mock_model.validate.return_value = None
    patched_bedrock_model.return_value = mock_model

    # Act
    result = await chat_completions(mock_chat_request)

    # Assert
    patched_bedrock_model.assert_called_once()
    mock_model.validate.assert_called_once_with(mock_chat_request)
    mock_model.chat_stream.assert_called_once_with(mock_chat_request)
    assert isinstance(result, StreamingResponse)
    assert result.media_type == "text/event-stream"
    # Don't compare generators directly as FastAPI wraps it in an async generator


@pytest.mark.asyncio
@patch("kuhl_haus.bedrock.api.routers.chat.BedrockModel")
@patch("kuhl_haus.bedrock.api.routers.chat.api_key_auth")
@patch("kuhl_haus.bedrock.api.routers.chat.Depends")
async def test_chat_completions_gpt_model_replacement(fastapi_dep, patched_api_key_auth, patched_bedrock_model, mock_chat_request):
    """Test that GPT models are replaced with the default model."""
    # Arrange
    mock_dep = MagicMock(spec=Depends)
    fastapi_dep.return_value = mock_dep
    api_auth = MagicMock()
    patched_api_key_auth.return_value = api_auth
    mock_chat_request.model = "gpt-4"
    mock_model = Mock()
    mock_response = Mock(spec=ChatResponse)
    mock_model.chat.return_value = mock_response
    mock_model.validate.return_value = None
    patched_bedrock_model.return_value = mock_model

    # Act
    await chat_completions(mock_chat_request)

    # Assert
    assert mock_chat_request.model == DEFAULT_MODEL
    mock_model.validate.assert_called_once_with(mock_chat_request)


@pytest.mark.asyncio
@patch("kuhl_haus.bedrock.api.routers.chat.BedrockModel")
@patch("kuhl_haus.bedrock.api.routers.chat.api_key_auth")
@patch("kuhl_haus.bedrock.api.routers.chat.Depends")
async def test_chat_completions_validation_failure(fastapi_dep, patched_api_key_auth, patched_bedrock_model, mock_chat_request):
    """Test that validation errors from the model are propagated."""
    # Arrange
    mock_dep = MagicMock(spec=Depends)
    fastapi_dep.return_value = mock_dep
    api_auth = MagicMock()
    patched_api_key_auth.return_value = api_auth
    mock_model = Mock()
    validation_error = ValueError("Unsupported model")
    mock_model.validate.side_effect = validation_error
    patched_bedrock_model.return_value = mock_model

    # Act & Assert
    with pytest.raises(ValueError) as excinfo:
        await chat_completions(mock_chat_request)

    assert str(excinfo.value) == "Unsupported model"
    mock_model.validate.assert_called_once_with(mock_chat_request)
    mock_model.chat.assert_not_called()


@pytest.mark.asyncio
@patch("kuhl_haus.bedrock.api.routers.chat.BedrockModel")
@patch("kuhl_haus.bedrock.api.routers.chat.api_key_auth")
@patch("kuhl_haus.bedrock.api.routers.chat.Depends")
async def test_chat_completions_chat_error(fastapi_dep, patched_api_key_auth, patched_bedrock_model, mock_chat_request):
    """Test handling of errors during chat generation."""
    # Arrange
    mock_dep = MagicMock(spec=Depends)
    fastapi_dep.return_value = mock_dep
    api_auth = MagicMock()
    patched_api_key_auth.return_value = api_auth
    mock_model = Mock()
    chat_error = RuntimeError("Chat generation failed")
    mock_model.validate.return_value = None
    mock_model.chat.side_effect = chat_error
    patched_bedrock_model.return_value = mock_model

    # Act & Assert
    with pytest.raises(RuntimeError) as excinfo:
        await chat_completions(mock_chat_request)

    assert str(excinfo.value) == "Chat generation failed"
    mock_model.validate.assert_called_once_with(mock_chat_request)
    mock_model.chat.assert_called_once_with(mock_chat_request)


@pytest.mark.asyncio
@patch("kuhl_haus.bedrock.api.routers.chat.BedrockModel")
@patch("kuhl_haus.bedrock.api.routers.chat.api_key_auth")
@patch("kuhl_haus.bedrock.api.routers.chat.Depends")
async def test_chat_completions_stream_error(fastapi_dep, patched_api_key_auth, patched_bedrock_model, mock_chat_request):
    """Test handling of errors during stream generation setup."""
    # Arrange
    mock_dep = MagicMock(spec=Depends)
    fastapi_dep.return_value = mock_dep
    api_auth = MagicMock()
    patched_api_key_auth.return_value = api_auth
    mock_chat_request.stream = True
    mock_model = Mock()
    stream_error = RuntimeError("Stream setup failed")
    mock_model.validate.return_value = None
    mock_model.chat_stream.side_effect = stream_error
    patched_bedrock_model.return_value = mock_model

    # Act & Assert
    with pytest.raises(RuntimeError) as excinfo:
        await chat_completions(mock_chat_request)

    assert str(excinfo.value) == "Stream setup failed"
    mock_model.validate.assert_called_once_with(mock_chat_request)
    mock_model.chat_stream.assert_called_once_with(mock_chat_request)
    mock_model.chat.assert_not_called()
