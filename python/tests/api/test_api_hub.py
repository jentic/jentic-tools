from typing import Any, Dict, List, Optional

import pytest
from pydantic import BaseModel

from jentic.api.api_hub import JenticAPIClient

# Assuming models are accessible via the path below
# Split imports to potentially resolve import errors
from jentic.models import AssociatedFiles, FileEntry, FileId, WorkflowEntry


# Minimal mock models for testing
class MockAssociatedFiles(AssociatedFiles):
    arazzo: List[FileId] = []  # Use FileId
    open_api: List[FileId] = []  # Use FileId


class MockFileEntry(FileEntry):
    pass


class MockWorkflowEntry(WorkflowEntry):
    # Override fields that expect complex types if not needed directly
    api_references: List[Any] = []  # Keep it simple for this test
    files: MockAssociatedFiles = MockAssociatedFiles()  # Use updated mock with defaults


@pytest.fixture
def api_client() -> JenticAPIClient:
    """Fixture to create a JenticAPIClient instance for testing."""
    # No need for real URLs or API keys for testing this method
    return JenticAPIClient()


# --- Test Cases --- #


def test_build_source_descriptions_happy_path(api_client):
    """Test URL-based matching of Arazzo OpenAPI sources to OpenAPI files."""
    workflow_entry = MockWorkflowEntry(
        workflow_id="wf1",
        workflow_uuid="uuid1",
        name="Test Workflow",
        files=MockAssociatedFiles(open_api=[FileId(id="file1_id"), FileId(id="file2_id")]),
        api_references=[],
    )
    # Content for the first available file (file1_id)
    content1 = {"openapi": "3.0", "info": {"title": "API One - First File"}}
    # Content for the second file (file2_id)
    content2 = {"openapi": "3.0", "info": {"title": "API Two - Second File"}}
    all_openapi_files = {
        "file1_id": MockFileEntry(
            id="file1_id", type="open_api", filename="api_one.json", content=content1, oak_path="./specs/api_one.json"
        ),
        "file2_id": MockFileEntry(
            id="file2_id", type="open_api", filename="api_two.yaml", content=content2, oak_path="./specs/api_two.yaml"
        ),
    }
    # Sources with URLs that match the filenames
    first_source_name = "ApiOneSourceFirst"
    second_source_name = "ApiTwoSourceSecond"
    arazzo_doc = {
        "sourceDescriptions": [
            {"name": first_source_name, "url": "./specs/api_one.json", "type": "openapi"},
            {"name": second_source_name, "url": "./specs/api_two.yaml", "type": "openapi"},
        ]
    }

    result = api_client._build_source_descriptions(
        workflow_entry=workflow_entry,
        all_openapi_files=all_openapi_files,
        arazzo_doc=arazzo_doc,
    )

    # Each source should map to the file with matching filename in the URL
    assert len(result) == 2
    assert first_source_name in result
    assert second_source_name in result
    assert result[first_source_name] == content1
    assert result[second_source_name] == content2


def test_build_source_descriptions_no_arazzo_sources(api_client):
    """Test when Arazzo doc has no sourceDescriptions. Should return empty dict."""
    workflow_entry = MockWorkflowEntry(
        workflow_id="wf1",
        workflow_uuid="uuid1",
        name="Test",
        files=MockAssociatedFiles(open_api=[FileId(id="file1_id")]),
    )
    all_openapi_files = {
        "file1_id": MockFileEntry(id="file1_id", type="open_api", filename="./api.json", content={})
    }
    arazzo_doc = {}  # No sourceDescriptions

    result = api_client._build_source_descriptions(workflow_entry, all_openapi_files, arazzo_doc)
    assert result == {}


def test_build_source_descriptions_empty_arazzo_sources(api_client):
    """Test when Arazzo doc has empty sourceDescriptions list. Should return empty dict."""
    workflow_entry = MockWorkflowEntry(
        workflow_id="wf1",
        workflow_uuid="uuid1",
        name="Test",
        files=MockAssociatedFiles(open_api=[FileId(id="file1_id")]),
    )
    all_openapi_files = {
        "file1_id": MockFileEntry(id="file1_id", type="open_api", filename="./api.json", content={})
    }
    arazzo_doc = {"sourceDescriptions": []}

    result = api_client._build_source_descriptions(workflow_entry, all_openapi_files, arazzo_doc)
    assert result == {}


def test_build_source_descriptions_missing_file_in_response(api_client):
    """Test when there's no URL match but a default content is available.
    Should use the default content (first available file content) as fallback.
    """
    # Workflow references 'missing_id' first, then 'file2_id'
    workflow_entry = MockWorkflowEntry(
        workflow_id="wf1",
        workflow_uuid="uuid1",
        name="Test",
        # The actual file IDs listed here ('missing_id', 'file2_id') are less critical
        # than what's in `all_openapi_files` for this specific test of matching logic.
        files=MockAssociatedFiles(open_api=[FileId(id="file2_id")]),
    )
    # Only the second file's content is available in the response
    content2 = {"info": "API Two - Content"}
    all_openapi_files = {
        "file2_id": MockFileEntry(
            id="file2_id", type="open_api", filename="api_two.json", content=content2, oak_path="./actual_path/api_two.json"
        )
        # 'missing_id' is not present here, which is fine for this test's purpose.
    }
    # Arazzo source with URL that doesn't match any oak_path
    arazzo_source_name = "ApiSourceNoMatch"
    arazzo_doc = {
        "sourceDescriptions": [
            {"name": arazzo_source_name, "url": "./specs/no_match.json", "type": "openapi"}
        ]
    }

    result = api_client._build_source_descriptions(
        workflow_entry=workflow_entry,
        all_openapi_files=all_openapi_files,
        arazzo_doc=arazzo_doc,
    )

    # With oak_path matching, if no exact match is found, the source is not included.
    # The previous fallback logic is removed.
    assert len(result) == 0
    assert arazzo_source_name not in result
