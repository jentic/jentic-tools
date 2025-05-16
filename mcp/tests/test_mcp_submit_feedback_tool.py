import os
from typing import Any

import pytest
import pytest_asyncio
import httpx
from unittest.mock import AsyncMock

from mcp.adapters.mcp import MCPAdapter
from mcp.handlers import handle_request


class MockStdioTransport:
    """Mock stdio transport for testing without any side effects."""

    def __init__(self, adapter):
        """Initialize the mock stdio transport."""
        self.adapter = adapter
        self.received_messages = []
        self.sent_responses = []

    async def send_response(self, response: dict[str, Any]) -> None:
        """Record sent responses."""
        self.sent_responses.append(response)

    async def process_message(self, message: dict[str, Any]) -> None:
        """Process a mock message."""
        self.received_messages.append(message)

        tool_name = message.get("type")
        data = message.get("data", {})

        if not tool_name:
            await self.send_response({"error": "Invalid message format: missing 'type'"})
            return

        try:
            result = await handle_request(tool_name, data) # Pass only tool_name and data
            await self.send_response(result) # Return result directly
        except ValueError as e:
            await self.send_response({"error": str(e), "result": {"success": False, "message": str(e)}})
        except Exception as e:
            error_info = f"Unhandled error in handle_request for {tool_name}: {str(e)}"
            # logger.error(error_info, exc_info=True) # logger is not defined
            await self.send_response({"error": error_info, "result": {"success": False, "message": error_info}})


@pytest_asyncio.fixture
async def mock_adapter():
    """Create an MCP adapter with mock components."""
    os.environ["MOCK_ENABLED"] = "true"
    temp_dir = os.path.join(os.getcwd(), ".test_output", "test_feedback_tool")
    os.makedirs(temp_dir, exist_ok=True)
    adapter = MCPAdapter()
    try:
        yield adapter
    finally:
        if "MOCK_ENABLED" in os.environ:
            del os.environ["MOCK_ENABLED"]


@pytest_asyncio.fixture
async def mcp_transport(mock_adapter):
    """Create a mock transport for testing."""
    transport = MockStdioTransport(adapter=mock_adapter)
    return transport


FEEDBACK_URL = "https://test.feedback.endpoint/dev/workflow-feedback"


@pytest_asyncio.fixture
def mock_feedback_env(mocker):
    """Mocks the environment variable for FEEDBACK_ENDPOINT_URL."""
    mocker.patch.dict(os.environ, {"FEEDBACK_ENDPOINT_URL": FEEDBACK_URL})


@pytest.mark.asyncio
async def test_submit_feedback_success(mcp_transport, mocker, mock_feedback_env):
    """Test successful feedback submission."""
    mock_post = mocker.patch("mcp.adapters.mcp.httpx.AsyncClient.post", new_callable=AsyncMock)
    
    mock_response_obj = mocker.MagicMock(spec=httpx.Response)
    mock_response_obj.status_code = 200
    mock_response_obj.raise_for_status = mocker.MagicMock() 
    mock_post.return_value = mock_response_obj

    feedback_data = {
        "uuid": "test-uuid",
        "error": "Test error",
        "context": "Testing feedback",
        "user_email": "test@example.com",
        "user_comments": "This is a test comment."
    }
    await mcp_transport.process_message(
        {"type": "submit_feedback", "data": {"feedback_data": feedback_data}}
    )

    assert len(mcp_transport.sent_responses) == 1
    response = mcp_transport.sent_responses[0]
    print("*" * 50 )
    print(response)
    print("*" * 50 )

    assert "result" in response, f"Response missing 'result': {response}"
    assert response["result"]["success"] is True, f"Expected success=True, got: {response['result']}"
    assert "Feedback submitted successfully" in response["result"].get("message", ""), f"Unexpected message: {response['result'].get('message')}"
    mock_post.assert_called_once_with(FEEDBACK_URL, json=feedback_data)


@pytest.mark.asyncio
async def test_submit_feedback_missing_data(mcp_transport, mock_feedback_env):
    """Test feedback submission with missing feedback_data."""
    await mcp_transport.process_message(
        {"type": "submit_feedback", "data": {}}
    )

    assert len(mcp_transport.sent_responses) == 1
    response = mcp_transport.sent_responses[0]
    assert response["result"]["success"] is False
    assert "Missing or invalid 'feedback_data' parameter" in response["result"]["message"]


@pytest.mark.asyncio
async def test_submit_feedback_invalid_data_type(mcp_transport, mock_feedback_env):
    """Test feedback submission with invalid type for feedback_data."""
    await mcp_transport.process_message(
        {"type": "submit_feedback", "data": {"feedback_data": "not_a_dict"}}
    )

    assert len(mcp_transport.sent_responses) == 1
    response = mcp_transport.sent_responses[0]
    assert response["result"]["success"] is False
    assert "Missing or invalid 'feedback_data' parameter" in response["result"]["message"]


@pytest.mark.asyncio
async def test_submit_feedback_network_error(mcp_transport, mocker, mock_feedback_env):
    """Test feedback submission with a httpx.RequestError."""
    mock_post = mocker.patch("mcp.adapters.mcp.httpx.AsyncClient.post", new_callable=AsyncMock)
    mock_request_obj = mocker.MagicMock(spec=httpx.Request)
    mock_post.side_effect = httpx.RequestError("Simulated network error", request=mock_request_obj)

    feedback_data = {"uuid": "test-uuid", "error": "Test error for network failure"}
    await mcp_transport.process_message(
        {"type": "submit_feedback", "data": {"feedback_data": feedback_data}}
    )

    assert len(mcp_transport.sent_responses) == 1
    response = mcp_transport.sent_responses[0]
    assert response["result"]["success"] is False
    assert "Failed to submit feedback due to network/request issue: Simulated network error" in response["result"]["message"]
    mock_post.assert_called_once_with(FEEDBACK_URL, json=feedback_data)


@pytest.mark.asyncio
async def test_submit_feedback_http_error(mcp_transport, mocker, mock_feedback_env):
    """Test feedback submission with an httpx.HTTPStatusError from the server."""
    mock_post = mocker.patch("mcp.adapters.mcp.httpx.AsyncClient.post", new_callable=AsyncMock)

    mock_returned_response = mocker.MagicMock(spec=httpx.Response)
    mock_returned_response.status_code = 500
    mock_returned_response.text = "Internal Server Error Text"

    error_response_obj = mocker.MagicMock(spec=httpx.Response)
    error_response_obj.status_code = 500
    error_response_obj.text = "Internal Server Error Text"
    error_request_obj = mocker.MagicMock(spec=httpx.Request)

    mock_returned_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        message="Server error 500", request=error_request_obj, response=error_response_obj
    )
    mock_post.return_value = mock_returned_response

    feedback_data = {"uuid": "test-uuid", "error": "Test error for HTTP failure"}
    await mcp_transport.process_message(
        {"type": "submit_feedback", "data": {"feedback_data": feedback_data}}
    )

    assert len(mcp_transport.sent_responses) == 1
    response = mcp_transport.sent_responses[0]
    assert response["result"]["success"] is False
    assert "Failed to submit feedback, server returned error: 500 - Internal Server Error Text" in response["result"]["message"]
    mock_post.assert_called_once_with(FEEDBACK_URL, json=feedback_data)


@pytest.mark.asyncio
async def test_submit_feedback_unexpected_error(mcp_transport, mocker, mock_feedback_env):
    """Test feedback submission with an unexpected error during submission logic."""
    mock_post = mocker.patch("mcp.adapters.mcp.httpx.AsyncClient.post", new_callable=AsyncMock)
    mock_post.side_effect = RuntimeError("Unexpected internal runtime issue")

    feedback_data = {"uuid": "test-uuid", "error": "Test error for unexpected failure"}
    await mcp_transport.process_message(
        {"type": "submit_feedback", "data": {"feedback_data": feedback_data}}
    )

    assert len(mcp_transport.sent_responses) == 1
    response = mcp_transport.sent_responses[0]
    assert response["result"]["success"] is False
    assert "An unexpected error occurred while submitting feedback: Unexpected internal runtime issue" in response["result"]["message"]
    mock_post.assert_called_once_with(FEEDBACK_URL, json=feedback_data)
