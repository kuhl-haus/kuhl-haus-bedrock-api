"""
    tests/conftest.py

    Read more about conftest.py under:
    - https://docs.pytest.org/en/stable/fixture.html
    - https://docs.pytest.org/en/stable/writing_plugins.html
"""

import os
import pytest
from unittest.mock import patch, MagicMock


# Define the mock_aws fixture that will be applied before any test runs
@pytest.fixture(autouse=True, scope="session")
def mock_aws():
    """
    Mock AWS services to prevent actual AWS API calls during tests.
    This is auto-used in all tests to avoid AWS credential errors.
    """
    # Create mock for SSM parameter store
    ssm_mock = MagicMock()
    ssm_mock.get_parameter.return_value = {
        'Parameter': {
            'Value': 'mock-api-key-secret'
        }
    }

    # Create mock for Bedrock
    bedrock_mock = MagicMock()
    bedrock_mock.list_foundation_models.return_value = {
        'modelSummaries': [
            {'modelId': 'anthropic.claude-3-sonnet-20240229-v1:0'},
            {'modelId': 'anthropic.claude-3-haiku-20240307-v1:0'},
            {'modelId': 'amazon.titan-text-express-v1'}
        ]
    }

    # Create mock for Secrets Manager
    secrets_manager_mock = MagicMock()
    secrets_manager_mock.get_secret_value.return_value = {
        'SecretString': '{"api_keys": ["test-api-key-1", "test-api-key-2"]}'
    }

    # Apply the mocks using patch
    patches = [
        patch('boto3.client', lambda service, **kwargs: {
            'ssm': ssm_mock,
            'bedrock-runtime': bedrock_mock,
            'bedrock': bedrock_mock,
            'secretsmanager': secrets_manager_mock
        }.get(service, MagicMock())),
    ]

    # Set mock environment variables needed for tests
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
    os.environ['SECRET_ARN_PARAMETER'] = '/bedrock/api/key/secret/arn'

    # Start all patches
    for p in patches:
        p.start()

    # Provide the fixture
    yield

    # Stop all patches
    for p in patches:
        p.stop()


# Add additional fixtures that might be needed for specific tests
@pytest.fixture
def mock_model_response():
    """Mock response from Bedrock model invocation"""
    return {
        "inputTokenCount": 10,
        "results": [{
            "outputText": "This is a mock response from the model.",
            "tokenCount": 15,
            "completionReason": "COMPLETED",
            "index": 0
        }]
    }


@pytest.fixture
def mock_embedding_response():
    """Mock response from Bedrock embedding model"""
    return {
        "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
        "inputTokenCount": 5
    }
