"""MCP adapter for the Jentic MCP Plugin."""

import logging
from dataclasses import asdict
from typing import Any
import os
import httpx

from jentic import Jentic
from jentic.models import ApiCapabilitySearchRequest, APISearchResults

from mcp.core.generators.code_generator import generate_code_sample


class MCPAdapter:
    """Model Configuration Protocol adapter for the Jentic MCP Plugin."""

    def __init__(self):
        """Initialize the MCP adapter."""
        self.jentic = Jentic()

    async def search_api_capabilities(self, request: dict[str, Any]) -> dict[str, Any]:
        """MCP endpoint for searching API capabilities.

        Args:
            request: MCP tool request parameters.

        Returns:
            MCP tool response.
        """
        request = ApiCapabilitySearchRequest(
            capability_description=request["capability_description"],
            keywords=request.get("keywords"),
            max_results=request.get("max_results", 5),
        )

        results: APISearchResults = await self.jentic.search_api_capabilities(request=request)

        # Dump results including api_name
        data = results.model_dump(exclude_none=False)
        
        # Extract API names from workflow API references
        data = self._ensure_api_names_in_response(data)
        
        # Prefix workflow summaries with their api_name
        for wf in data.get("workflows", []):
            api = wf.get("api_name")
            if api:
                # Prefix with full api_name (keep the dot)
                wf_summary = wf.get("summary", "")
                wf["summary"] = f"{api}-{wf_summary}"
        return {
            "result": {
                "matches": data,
                "query": request.capability_description,
                "total_matches": len(results.workflows) + len(results.operations),
            }
        }

    def _extract_api_name_from_refs(self, workflow: dict[str, Any]) -> str | None:
        """Extract API name from workflow references.
        
        Args:
            workflow: A workflow dictionary.
            
        Returns:
            The extracted API name if available, None otherwise.
        """
        api_refs = workflow.get("api_references", [])
        if not api_refs or not isinstance(api_refs, list) or not api_refs:
            return None
            
        first_ref = api_refs[0]
        if isinstance(first_ref, dict) and "api_name" in first_ref:
            return first_ref["api_name"]
        return None

    def _ensure_api_names_in_response(self, response_data: dict[str, Any]) -> dict[str, Any]:
        """Extract API names from workflow API references and add them to the workflow configuration.
        
        Args:
            response_data: The response data from the Jentic API.
            
        Returns:
            The response data with API names added to workflows.
        """
        # Handle different formats of workflows (list in search results, dict in config)
        workflows = response_data.get("workflows", {})
        
        # For list format (search_api_capabilities response)
        if isinstance(workflows, list):
            for wf in workflows:
                if isinstance(wf, dict) and "api_name" not in wf:
                    api_name = self._extract_api_name_from_refs(wf)
                    if api_name:
                        wf["api_name"] = api_name
        
        # For dict format (generate_runtime_config response)
        elif isinstance(workflows, dict):
            for wf_uuid, wf_conf in workflows.items():
                if "api_name" not in wf_conf:
                    api_name = self._extract_api_name_from_refs(wf_conf)
                    if api_name:
                        wf_conf["api_name"] = api_name
                    
        return response_data

    async def generate_runtime_config(self, request: dict[str, Any]) -> dict[str, Any]:
        """MCP endpoint for generating a configuration file from a selection set.

        Args:
            request: MCP tool request parameters.

        Returns:
            MCP tool response.
        """
        # Get the workflow and operation UUIDs from the request
        workflow_uuids = request.get("workflow_uuids", [])
        if isinstance(workflow_uuids, str):
            workflow_uuids = [workflow_uuids]
        operation_uuids = request.get("operation_uuids", [])
        if isinstance(operation_uuids, str):
            operation_uuids = [operation_uuids]
        
        # Get the API name or use empty string as default
        api_name = request.get("api_name", "")

        logger = logging.getLogger(__name__)
        logger.debug(
            f"Generating config with workflow_uuids: {workflow_uuids}, operation_uuids: {operation_uuids}, api_name: {api_name}"
        )

        try:
            # Generate configuration from the selection set
            result = await self.jentic.load_execution_info(
                workflow_uuids=workflow_uuids, operation_uuids=operation_uuids, api_name=api_name
            )
            return {"result": result}
        except ValueError as e:
            logger.error(f"Error generating config: {str(e)}")
            return {
                "result": {
                    "success": False,
                    "operation_uuids": operation_uuids,
                    "workflow_uuids": workflow_uuids,
                    "api_name": api_name,
                    "message": str(e),
                    "config": {},
                }
            }

    async def generate_code_sample(self, request: dict[str, Any]) -> dict[str, Any]:
        """MCP endpoint for generating code samples.

        Args:
            request: MCP tool request parameters.

        Returns:
            MCP tool response with generated code sample.
        """
        format_name = request.get("format")
        language = request.get("language")

        logger = logging.getLogger(__name__)
        logger.debug(f"Generating code sample for format: {format_name}, language: {language}")

        try:
            # Generate the code sample
            code = generate_code_sample(format=format_name, language=language)

            return {"result": {"success": True, "code": code}}
        except Exception as e:
            logger.error(f"Error generating code sample: {str(e)}")
            return {"result": {"success": False, "message": str(e)}}

    def get_execute_tool_failure_suggested_next_actions(self) -> list[dict[str, str]]:
        """Helper function to provide suggested next actions for error handling."""
        return [
            {
                "tool_name": "submit_feedback",
                "description": (
                    "Ask the user for permission to submit feedback, which includes reporting details of the error to Jentic for analysis."
                    "Before calling the submit_feedback tool, always show the user the full feedback content that will be sent."
                    "Ensure that NO SENSITIVE information (e.g., API keys, access tokens) is present in the feedback. If such information is detected in the error message, REMOVE it before proceeding."
                    "Include the error message JSON from the execute tool call response in the 'error' field of the submit_feedback tool call — only after confirming it contains NO SENSITIVE DATA, If such information is detected in the error message, REMOVE it before proceeding."
                    "Prompt the user to optionally provide: "
                    "Their email address (for follow-up once Jentic reviews the issue) and "
                    "any additional comments they wish to include in the feedback."
                )
            }
        ]

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        """MCP endpoint for executing an operation or workflow.

        Args:
            params: MCP tool request parameters containing execution_type, uuid, and inputs.

        Returns:
            MCP tool response with the execution result.
        """
        logger = logging.getLogger(__name__)
        execution_type = params.get("execution_type")
        uuid = params.get("uuid")
        inputs = params.get("inputs", {})

        if not execution_type or execution_type not in ["operation", "workflow"]:
            logger.error(f"Invalid execution_type: {execution_type}")
            return {"result": {"success": False, "message": "Invalid execution_type. Must be 'operation' or 'workflow'."}}
        if not uuid:
            logger.error("Missing required parameter: uuid")
            return {"result": {"success": False, "message": "Missing required parameter: uuid"}}
        if not isinstance(inputs, dict):
             logger.error(f"Invalid inputs type: {type(inputs)}. Must be a dictionary.")
             return {"result": {"success": False, "message": "Invalid inputs type. Must be a dictionary."}}

        logger.info(f"Executing {execution_type} with uuid: {uuid} and inputs: {inputs}")

        try:
            if execution_type == "operation":
                result = await self.jentic.execute_operation(operation_uuid=uuid, inputs=inputs)
                # Operations typically return a dict
                return {"result": {"success": True, "output": result}}
            elif execution_type == "workflow":
                result = await self.jentic.execute_workflow(workflow_uuid=uuid, inputs=inputs)
                # Check the success value in the WorkflowResult
                if hasattr(result, 'success') and not result.success:
                    return {
                        "result": {
                                "success": False,
                                 "message": result.error or "Workflow execution failed.",
                                 "output": asdict(result),
                                 "suggested_next_actions": self.get_execute_tool_failure_suggested_next_actions()
                             }
                    }
                return {"result": {"success": True, "output": asdict(result)}}

        except Exception as e:
            logger.error(f"Error executing {execution_type} {uuid}: {str(e)}", exc_info=True)
            return {
                        "result": {
                            "success": False,
                            "message": f"Error during execution: {str(e)}",
                            "suggested_next_actions": self.get_execute_tool_failure_suggested_next_actions()
                        }
                    }

    async def submit_feedback(self, params: dict[str, Any]) -> dict[str, Any]:
        """MCP endpoint for submitting feedback, typically about a failed execution.
            Makes a http call to a Jentic endpoint to submit the feedback

        Args:
            params: MCP tool request parameters. Expected to contain:
                    - 'feedback_data': A dictionary with the feedback content.

        Returns:
            MCP tool response indicating success or failure of feedback submission.
        """
        logger = logging.getLogger(__name__)
        feedback_data = params.get("feedback_data")

        if not feedback_data or not isinstance(feedback_data, dict):
            logger.error("Missing or invalid 'feedback_data' in request for submit_feedback")
            return {"result": {"success": False, "message": "Missing or invalid 'feedback_data' parameter."}}

        # Source user JENTIC_UUID from environment to pass in the feedback.
        env_jentic_uuid = os.environ.get("JENTIC_UUID")
        if env_jentic_uuid:
            feedback_data['jentic_uuid'] = env_jentic_uuid
            logger.info(f"JENTIC_UUID ('{env_jentic_uuid}') from environment ensured in feedback_data.")
        else:
            logger.debug("JENTIC_UUID not found in environment. It will not be included in the feedback.")

        feedback_endpoint_url = os.environ.get("FEEDBACK_ENDPOINT_URL", "https://xze2r4egy7.execute-api.eu-west-1.amazonaws.com/dev/workflow-feedback")
        logger.info(f"Submitting feedback to {feedback_endpoint_url}: {feedback_data}")

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(feedback_endpoint_url, json=feedback_data)
                response.raise_for_status()
            logger.info(f"Feedback submitted successfully. Response: {response.status_code}")
            return {"result": {"success": True, "message": "Feedback submitted successfully. The Jentic team will look into it and get back to you at the submitted email if provided."}}
        except httpx.RequestError as e:
            logger.error(f"Error submitting feedback (network/request issue): {str(e)}", exc_info=True)
            return {
                "result": {
                    "success": False,
                    "message": f"Failed to submit feedback due to network/request issue: {str(e)}",
                }
            }
        except httpx.HTTPStatusError as e:
            logger.error(f"Error submitting feedback (HTTP status): {str(e)} - Response: {e.response.text}", exc_info=True)
            return {
                "result": {
                    "success": False,
                    "message": f"Failed to submit feedback, server returned error: {e.response.status_code} - {e.response.text}",
                }
            }
        except Exception as e:
            logger.error(f"An unexpected error occurred while submitting feedback: {str(e)}", exc_info=True)
            return {
                "result": {
                    "success": False,
                    "message": f"An unexpected error occurred while submitting feedback: {str(e)}",
                }
            }
