import pytest
import pytest_asyncio
import os
import json
from typing import Dict, Any
from pathlib import Path

from jentic.jentic import Jentic
from jentic.agent_runtime.tool_execution import WorkflowResult
from jentic.models import ApiCapabilitySearchRequest, APISearchResults


def get_discord_operation_uuid(target_env: str, env_file_path: Path) -> str:
    """Get the Discord operation UUID from environment variables."""
    uuid = os.getenv("DISCORD_GET_MY_USER_OPERATION_UUID")
    if not uuid or "your_" in uuid:  # Check for placeholder
        pytest.fail(f"DISCORD_GET_MY_USER_OPERATION_UUID not set or is placeholder in {env_file_path} for {target_env} environment")
    return uuid

def get_discord_workflow_uuid(target_env: str, env_file_path: Path) -> str:
    """Get the Discord workflow UUID from environment variables."""
    uuid = os.getenv("DISCORD_GET_USER_DETAILS_WORKFLOW_UUID")
    if not uuid or "your_" in uuid:  # Check for placeholder
        pytest.fail(f"DISCORD_GET_USER_DETAILS_WORKFLOW_UUID not set or is placeholder in {env_file_path} for {target_env} environment")
    return uuid


@pytest_asyncio.fixture
async def loaded_jentic_execution_info(target_env: str, env_file_path: Path) -> Dict[str, Any]:
    """Fixture to load execution info for a standard Discord operation and workflow.
    Expects exec_info['operations'] and exec_info['workflows'] to be dicts keyed by UUID.
    """
    jentic_client = Jentic()
    op_uuid = get_discord_operation_uuid(target_env, env_file_path)
    wf_uuid = get_discord_workflow_uuid(target_env, env_file_path)
    exec_info = await jentic_client.load_execution_info(
        operation_uuids=[op_uuid],
        workflow_uuids=[wf_uuid]
    )
    # Basic validation of fixture data itself
    assert "operations" in exec_info, "Fixture: 'operations' key missing"
    assert isinstance(exec_info["operations"], dict), "Fixture: operations should be a dict"
    assert len(exec_info["operations"]) == 1, "Fixture: Expected 1 operation in operations dict"
    assert op_uuid in exec_info["operations"], f"Fixture: op_uuid {op_uuid} not in operations keys"
    assert exec_info["operations"][op_uuid].get("operation_uuid") == op_uuid, "Fixture: Operation UUID mismatch in details"

    assert "workflows" in exec_info, "Fixture: 'workflows' key missing"
    assert isinstance(exec_info["workflows"], dict), "Fixture: workflows should be a dict"
    assert len(exec_info["workflows"]) == 1, \
        f"Fixture: Expected 1 workflow in workflows dict, got {len(exec_info['workflows'])}. Keys: {list(exec_info['workflows'].keys())}"

    # exec_info["workflows"] is keyed by workflow name, not UUID.
    # Retrieve the single workflow's details (assuming only one was requested and returned).
    workflow_details_list = list(exec_info["workflows"].values())
    # This assertion is slightly redundant given the len check above, but good for clarity.
    assert len(workflow_details_list) == 1, "Expected one workflow detail object in the list of values."
    workflow_detail = workflow_details_list[0]
    
    assert workflow_detail.get("workflow_uuid") == wf_uuid, \
        f"Fixture: Workflow UUID mismatch in details. Expected {wf_uuid}, got {workflow_detail.get('workflow_uuid')}"
    return exec_info


@pytest.mark.asyncio
async def test_load_execution_info(loaded_jentic_execution_info: Dict[str, Any]):
    """Test loading execution information for operations and workflows using fixture."""
    exec_info = loaded_jentic_execution_info

    assert isinstance(exec_info, dict), "Expected execution info to be a dict"
    assert "operations" in exec_info, "'operations' key missing in exec_info"
    assert "workflows" in exec_info, "'workflows' key missing in exec_info"
    
    assert isinstance(exec_info["operations"], dict), "'operations' should be a dict"
    assert isinstance(exec_info["workflows"], dict), "'workflows' should be a dict"
    
    assert len(exec_info["operations"]) >= 1, "Expected at least one operation"
    assert len(exec_info["workflows"]) >= 1, "Expected at least one workflow"

    # Check presence of correct uuid key in the items
    first_op_uuid_key = list(exec_info["operations"].keys())[0]
    first_op_details = exec_info["operations"].get(first_op_uuid_key, {})
    assert "operation_uuid" in first_op_details, "'operation_uuid' missing in operation details"

    first_wf_uuid_key = list(exec_info["workflows"].keys())[0]
    first_wf_details = exec_info["workflows"].get(first_wf_uuid_key, {})
    assert "workflow_uuid" in first_wf_details, "'workflow_uuid' missing in workflow details"


@pytest.mark.asyncio
async def test_execute_discord_get_my_user_operation(target_env: str, env_file_path: Path):
    """Test executing a Discord read-only operation based on environment."""
    operation_uuid = get_discord_operation_uuid(target_env, env_file_path)
    jentic_client = Jentic()

    try:
        result = await jentic_client.execute_operation(
            operation_uuid=operation_uuid, inputs={}
        )
        assert isinstance(result, Dict), f"Expected result to be a Dict, got {type(result)}"
    except Exception as e:
        pytest.fail(f"execute_operation raised an exception: {e}")


@pytest.mark.asyncio
async def test_execute_discord_get_user_details_workflow(target_env: str, env_file_path: Path):
    """Test executing a Discord read-only workflow based on environment."""
    workflow_uuid = get_discord_workflow_uuid(target_env, env_file_path)
    jentic_client = Jentic()

    try:
        result = await jentic_client.execute_workflow(
            workflow_uuid=workflow_uuid, inputs={}
        )
        assert isinstance(result, WorkflowResult), f"Expected result to be a WorkflowResult, got {type(result)}"
    except Exception as e:
        pytest.fail(f"execute_workflow raised an exception: {e}")


@pytest.mark.asyncio
async def test_search_api_capabilities():
    """Test searching for API capabilities."""
    jentic_client = Jentic()
    request = ApiCapabilitySearchRequest(capability_description="discord user details", max_results=1)
    try:
        result = await jentic_client.search_api_capabilities(request)
        assert isinstance(result, APISearchResults), f"Expected APISearchResults, got {type(result)}"
        assert isinstance(result.operations, list), "Expected result.operations to be a list"
        assert isinstance(result.workflows, list), "Expected result.workflows to be a list"
        
        # Check that we have at least one result in either operations or workflows
        assert len(result.operations) > 0 or len(result.workflows) > 0, "Expected at least one search result in operations or workflows"

        # Optional: further checks if the content is predictable enough
        # For example, if expecting an operation:
        # if result.operations:
        #     assert "discord" in result.operations[0].name.lower(), "Expected a discord related item in operations"
        # elif result.workflows:
        #     assert "discord" in result.workflows[0].name.lower(), "Expected a discord related item in workflows"

    except Exception as e:
        pytest.fail(f"search_api_capabilities raised an exception: {e}")


@pytest.mark.asyncio
async def test_generate_llm_tool_definitions(
    tmp_path: Path, 
    loaded_jentic_execution_info: Dict[str, Any],
):
    """Test generating LLM tool definitions using loaded execution info."""
    jentic_client = Jentic()
    
    config_file = tmp_path / "jentic.json"
    with open(config_file, 'w') as f:
        json.dump(loaded_jentic_execution_info, f)

    try:
        definitions = jentic_client.generate_llm_tool_definitions(format="openai", config_path=str(config_file))
        assert isinstance(definitions, list), "Expected a list of tool definitions"
        assert len(definitions) > 0, "Expected at least one tool definition"
        tool_def = definitions[0]
        assert isinstance(tool_def, dict), "Expected tool definition to be a dict"
        assert tool_def.get("type") == "function"
        assert "function" in tool_def
        assert tool_def["function"].get("name"), "Tool function should have a name"
    except Exception as e:
        pytest.fail(f"generate_llm_tool_definitions (openai) raised an exception: {e}")

    # Test ValueError for empty format
    with pytest.raises(ValueError, match="format must be specified"):
        jentic_client.generate_llm_tool_definitions(format="", config_path=str(config_file))


@pytest.mark.asyncio
async def test_run_llm_tool(
    tmp_path: Path,
    loaded_jentic_execution_info: Dict[str, Any],
):
    """Test running an LLM tool, using loaded execution info for definition generation."""
    jentic_client = Jentic()

    # Test 1: Attempt to run tool before definitions are generated
    tool_not_loaded_result = await jentic_client.run_llm_tool("any_tool_name")
    assert isinstance(tool_not_loaded_result, dict)
    assert tool_not_loaded_result.get("success") is False
    assert "No tools found" in tool_not_loaded_result.get("error", "")

    # Test 2: Generate definitions and then run a tool
    # Use the full config loaded from the API
    config_file = tmp_path / "jentic_for_run_test.json"
    with open(config_file, 'w') as f:
        json.dump(loaded_jentic_execution_info, f)
        
    # Get the API-generated tool name for later use
    
    # Generate tool definitions from the config file
    definitions = jentic_client.generate_llm_tool_definitions(format="openai", config_path=str(config_file))
    assert definitions, "Expected non-empty tool definitions"
    
    # Find the specific Discord tools we want to test
    tool_names = []
    for tool_def in definitions:
        if "function" in tool_def:
            name = tool_def["function"].get("name")
            if name in ["get-authenticated-user-details", "get-users-me"]:
                tool_names.append(name)
    
    # We need both tools for comprehensive testing
    assert "get-users-me" in tool_names, "Operation 'get-users-me' not found in available tools"
    assert "get-authenticated-user-details" in tool_names, "Workflow 'get-authenticated-user-details' not found in available tools"
    
    # Test 1: Run the operation (direct API call)
    print("\nTesting operation: get-users-me")
    operation_result = await jentic_client.run_llm_tool(tool_name="get-users-me", inputs={})
    
    # Operation should return a direct dict response
    assert isinstance(operation_result, dict), f"Operation expected to return dict, got {type(operation_result)}"
    # The Discord API returns user ID directly
    assert "id" in operation_result, "Discord user ID not found in operation result"
    
    # Test 2: Run the workflow
    print("\nTesting workflow: get-authenticated-user-details")
    workflow_result = await jentic_client.run_llm_tool(tool_name="get-authenticated-user-details", inputs={})
    
    # Workflow should return a WorkflowResult
    assert isinstance(workflow_result, WorkflowResult), f"Workflow expected to return WorkflowResult, got {type(workflow_result)}"
    assert workflow_result.success, f"Workflow execution failed: {workflow_result.error}"
    
    # The workflow returns an email object with ID inside
    assert isinstance(workflow_result.output, dict), "Expected dict output from workflow"
    assert "email" in workflow_result.output, "User email data not found in workflow output"
    assert "id" in workflow_result.output["email"], "User ID not found in workflow email data"
