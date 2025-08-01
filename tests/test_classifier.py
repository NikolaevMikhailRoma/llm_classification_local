import pytest
from unittest.mock import MagicMock, patch
from src.classifier import MessageClassifier

@pytest.fixture
def classifier():
    """Provides a MessageClassifier instance with a mocked client."""
    with patch('src.classifier.OpenAI') as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        return MessageClassifier(base_url="http://fake-url", api_key="fake-key")

def test_classify_message(classifier):
    """Test that classify_message correctly calls the LLM with a list of messages."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "  category1,  category2 "
    classifier.client.chat.completions.create.return_value = mock_response

    messages = [
        {"role": "system", "content": "System instructions"},
        {"role": "user", "content": "User message"}
    ]

    result = classifier.classify_message(messages)

    assert result == ["category1", "category2"]
    classifier.client.chat.completions.create.assert_called_once_with(
        model="local-model",
        messages=messages,
        temperature=0.1
    )

def test_classify_with_api_error(classifier):
    """Tests that the method handles API errors gracefully."""
    classifier.client.chat.completions.create.side_effect = Exception("API Error")
    
    result = classifier.classify_message([{"role": "user", "content": "..."}])

    assert result == ["Error: Classification failed"]
