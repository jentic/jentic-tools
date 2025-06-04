"""LLM Tool specification library for different AI platforms."""

import logging
import re
from typing import Any, Literal

from .config import JenticConfig

logger = logging.getLogger(__name__)

# Constants for tool schema properties
OPENAI_FUNCTION_SCHEMA = {
    "type": "function",
    "function": {
        "name": "",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

ANTHROPIC_TOOL_SCHEMA = {
    "name": "",
    "description": "",
    "input_schema": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
}


class LLMToolSpecManager:
    """Dynamic tool specification manager for LLM tool-calling integrations."""

    def __init__(self) -> None:
        """Initialize the LLM tool specification manager."""
        self._workflow_definitions: dict[str, dict[str, Any]] = {}
        self._operation_definitions: dict[str, dict[str, Any]] = {}
        self._operation_name_to_uuid: dict[str, str] = {}
        self._workflow_name_to_uuid: dict[str, str] = {}
        self._tool_specs: dict[Literal["openai", "anthropic"], Any | None] = {
            "openai": None,
            "anthropic": None,
        }
        # Mapping of tool_name -> {sanitized_name: original_name}
        self._parameter_mappings: dict[str, dict[str, str]] = {}

        # Pre-compiled regex for validating parameter names across LLM platforms
        # This follows Anthropic's more restrictive pattern (alphanumeric, underscore, hyphen)
        # which is also compatible with OpenAI's more permissive schema
        self._valid_param_name_pattern = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
        # Vendor prefixes are added when api_name is present

    def load_workflows(self, workflows: dict[str, Any]) -> None:
        """Load workflow specifications into the manager.

        Args:
            workflows: Dictionary of Arazzo workflow specifications.
        """
        self._workflow_definitions.update(workflows)
        # Populate workflow name -> uuid mapping
        for workflow_name, workflow in workflows.items():
            try:
                workflow_uuid = workflow.get("workflow_uuid")
            except:
                logger.warning(
                    f"Could not find UUID for workflow tool name {workflow_name}, skipping mapping."
                )
                continue
            self._workflow_name_to_uuid[workflow_name] = workflow_uuid
            logger.debug(f"Mapping workflow tool name '{workflow_name}' to UUID '{workflow_uuid}'")

        # Reset cached specs
        self._tool_specs["openai"] = None
        self._tool_specs["anthropic"] = None

    def load_operations(self, operations: dict[str, Any]) -> None:
        """Load operation specifications into the manager.

        Args:
            operations: Dictionary of Jentic operation specifications.
        """
        logger.info(f"Loading {len(operations)} operation definitions.")
        # Store the raw definitions
        self._operation_definitions.update(operations)
        # Create the name -> uuid mapping
        for op_uuid, op_def in operations.items():
            op_name = self._generate_operation_tool_name(op_def)
            if op_name:
                self._operation_name_to_uuid[op_name] = op_uuid
                logger.debug(f"Mapping operation tool name '{op_name}' to UUID '{op_uuid}'")
            else:
                logger.warning(
                    f"Could not generate tool name for operation UUID {op_uuid}, skipping mapping."
                )

        # Reset cached specs
        self._tool_specs["openai"] = None
        self._tool_specs["anthropic"] = None

    def load_from_jentic_config(self, config: JenticConfig) -> None:
        """Load workflows and operations directly from a JenticConfig object."""
        logger.info("Loading tools from JenticConfig object.")
        workflows: dict[str, Any] = {}
        operations: dict[str, Any] = {}

        try:
            workflows = config.get_workflows()
            operations = config.get_operations()
            logger.debug(
                f"Loaded {len(workflows)} workflows and {len(operations)} operations via JenticConfig getter methods."
            )
        except Exception as e:
            logger.warning(
                "Could not retrieve workflows or operations from the provided JenticConfig."
            )

        self.load_workflows(workflows)
        self.load_operations(operations)

    def get_tool_specs(self, format: Literal["openai", "anthropic"] = "openai") -> dict[str, Any]:
        """Get tool specifications in the requested format.

        Args:
            format: The format to return ("openai" or "anthropic")

        Returns:
            Tool specifications in the requested format

        Raises:
            ValueError: If the format is not supported
        """
        if format.lower() == "openai":
            if not self._tool_specs["openai"]:
                self._tool_specs["openai"] = self._create_openai_tool_specs()
            return self._tool_specs["openai"]
        elif format.lower() == "anthropic":
            if not self._tool_specs["anthropic"]:
                self._tool_specs["anthropic"] = self._create_anthropic_tool_specs()
            return self._tool_specs["anthropic"]
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _create_openai_tool_specs(self) -> dict[str, Any]:
        """Create tool specifications for OpenAI function calling.

        Returns:
            Dictionary with tool specifications.
        """
        tools = []

        for workflow_id, workflow in self._workflow_definitions.items():
            function_schema = self._create_openai_function_schema(workflow_id, workflow)
            tools.append(
                {
                    "type": "function",
                    "function": function_schema,
                }
            )

        for operation_uuid, operation in self._operation_definitions.items():
            function_schema = self._create_openai_operation_schema(operation_uuid, operation)
            tools.append(
                {
                    "type": "function",
                    "function": function_schema,
                }
            )

        return {
            "tools": tools,
            "format": "openai",
        }

    def _create_openai_function_schema(
        self, workflow_id: str, workflow: dict[str, Any]
    ) -> dict[str, Any]:
        """Create a function schema for an OpenAI tool.

        Args:
            workflow_id: ID of the workflow.
            workflow: Arazzo workflow specification.

        Returns:
            OpenAI function schema.
        """
        parameters = self._extract_parameters(workflow)
        required = self._extract_required_parameters(workflow)
        
        # Sanitize parameter names for consistency
        sanitized_parameters, sanitized_required = self._sanitize_parameters(
            workflow_id, parameters, required
        )
        
        name = workflow_id
        if "api_name" in workflow:
            # Only add prefix if api_name is explicitly provided
            vendor = self._sanitize_vendor_name(workflow["api_name"])
            name = f"{vendor}-{workflow_id}"
        
        return {
            "name": name,
            "description": workflow.get("description", f"Execute the {workflow_id} workflow"),
            "parameters": {
                "type": "object",
                "properties": sanitized_parameters,
                "required": sanitized_required,
            },
        }

    def _create_openai_operation_schema(
        self, operation_uuid: str, operation: dict[str, Any]
    ) -> dict[str, Any]:
        """Create a function schema for an OpenAI tool based on an operation.

        Args:
            operation_uuid: Internal ID of the operation (not used for name).
            operation: Jentic operation specification.

        Returns:
            OpenAI function schema.
        """
        tool_name = self._generate_operation_tool_name(operation)
        parameters, required = self._extract_operation_parameters(operation)
        description = (
            operation.get("summary")
            or operation.get("description")
            or f"Execute {operation.get('method', 'HTTP')} request to {operation.get('path', 'endpoint')}"
        )
        
        name = tool_name

        return {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        }

    def _create_anthropic_tool_specs(self) -> dict[str, Any]:
        """Create tool specifications for Anthropic's Claude.

        Returns:
            Dictionary with tool specifications.
        """
        tools = []

        for workflow_id, workflow in self._workflow_definitions.items():
            tool_schema = self._create_anthropic_tool_schema(workflow_id, workflow)
            tools.append(tool_schema)

        for operation_uuid, operation in self._operation_definitions.items():
            tool_schema = self._create_anthropic_operation_schema(operation_uuid, operation)
            tools.append(tool_schema)

        return {
            "tools": tools,
            "format": "anthropic",
        }

    def _create_anthropic_tool_schema(
        self, workflow_id: str, workflow: dict[str, Any]
    ) -> dict[str, Any]:
        """Create a tool schema for Anthropic's Claude.

        Args:
            workflow_id: ID of the workflow.
            workflow: Arazzo workflow specification.

        Returns:
            Anthropic tool schema.
        """
        parameters = self._extract_parameters(workflow)
        required = self._extract_required_parameters(workflow)
        
        name = workflow_id
        if "api_name" in workflow:
            # Only add prefix if api_name is explicitly provided
            vendor = self._sanitize_vendor_name(workflow["api_name"])
            name = f"{vendor}-{workflow_id}"

        # Sanitize parameter keys
        sanitized_parameters, sanitized_required = self._sanitize_parameters(
            workflow_id, parameters, required
        )

        return {
            "name": name,
            "description": workflow.get("description", f"Execute the {workflow_id} workflow"),
            "input_schema": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": sanitized_parameters,
                "required": sanitized_required,
                "additionalProperties": False,
            },
        }

    def _create_anthropic_operation_schema(
        self, operation_uuid: str, operation: dict[str, Any]
    ) -> dict[str, Any]:
        """Create a tool schema for Anthropic's Claude based on an operation.

        Args:
            operation_uuid: Internal ID of the operation (not used for name).
            operation: Jentic operation specification.

        Returns:
            Anthropic tool schema.
        """
        tool_name_base = self._generate_operation_tool_name(operation)
        
        # Convert to Anthropic's preferred kebab-case
        tool_name = tool_name_base.replace("_", "-").lower()
        
        # Only add prefix if api_name is explicitly provided and valid
        if "api_name" in operation:
            vendor = self._sanitize_vendor_name(operation["api_name"])
            if vendor:
                tool_name = f"{vendor}-{tool_name}"

        parameters, required = self._extract_operation_parameters(operation)
        description = (
            operation.get("summary")
            or operation.get("description")
            or f"Execute {operation.get('method', 'HTTP')} request to {operation.get('path', 'endpoint')}"
        )

        # Ensure required is a list for the schema
        required_list = list(required) if required else []

        # Correctly format parameters for Anthropic schema (remove inline 'required')
        formatted_parameters = {}
        for param_name, param_details in parameters.items():
            # Create a copy to avoid modifying the original dict if needed elsewhere
            clean_details = param_details.copy()
            # Remove the incorrect 'required' flag if present
            clean_details.pop("required", None)
            formatted_parameters[param_name] = clean_details

        # Sanitize parameter keys
        sanitized_parameters, sanitized_required = self._sanitize_parameters(
            tool_name, formatted_parameters, required_list
        )

        return {
            "name": tool_name,
            "description": description,
            "input_schema": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": sanitized_parameters,
                "required": sanitized_required,
                "additionalProperties": False,
            },
        }

    def _clean_path_for_tool_name(self, path: str) -> str:
        """Cleans a URL path for use in a tool name.

        - Removes leading/trailing slashes.
        - Replaces internal slashes with hyphens.
        - Removes parameter braces {}.
        - Removes non-alphanumeric characters (except hyphens).
        """
        if not path:
            return ""
        cleaned = path.strip("/")
        cleaned = cleaned.replace("/", "-")  # Use hyphens
        cleaned = re.sub(r"\{([^}]+)\}", r"\1", cleaned)
        cleaned = re.sub(r"[^a-zA-Z0-9-]", "", cleaned)  # Allow hyphens
        cleaned = re.sub(r"-+", "-", cleaned).strip("-")  # Consolidate hyphens
        return cleaned

    def _generate_workflow_tool_name(self, workflow_uuid: str, workflow_def: dict[str, Any]) -> str:
        """Generate a tool name for a workflow, preferring 'name' if available."""
        # Prefer 'name' if explicitly defined in the workflow definition
        name = workflow_def.get("name")
        if name:
            return name
        # Fallback to workflow_uuid if 'name' is not present
        return workflow_uuid

    def _generate_operation_tool_name(self, operation_def: dict[str, Any]) -> str | None:
        """Generates a name for an operation based on the HTTP method and path.

        Args:
            operation_def: Dictionary with operation definition.

        Returns:
            Generated name for the operation or None if not enough information.
        """
        method = operation_def.get("method", "").lower()
        path = operation_def.get("path", "")
        cleaned_path = self._clean_path_for_tool_name(path)
        
        if method and cleaned_path:
            base = f"{method}-{cleaned_path}"
        elif cleaned_path:
            base = cleaned_path
        else:
            base = operation_def.get("operation_uuid", "unknown_operation")

        # Vendor prefix logic
        if "api_name" in operation_def:
            vendor = self._sanitize_vendor_name(operation_def["api_name"])
            # Check if base already starts with vendor prefix
            already_prefixed = base.startswith(f"{vendor}-")
            
            # Only add prefix if it doesn't already start with the vendor name
            if vendor and not already_prefixed:
                name = f"{vendor}-{base}"
                return name
                
        return base


    def _extract_parameters(self, definition: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Extract parameters from a workflow definition.

        Args:
            definition: Arazzo workflow specification.

        Returns:
            Dictionary of parameter schemas.
        """
        parameters = {}

        if "inputs" not in definition:
            return parameters

        inputs = definition["inputs"]

        if isinstance(inputs, dict) and "$ref" in inputs:
            return {
                "input": {
                    "type": "object",
                    "description": f"Input for {definition.get('workflowId', 'workflow')}",
                }
            }

        if isinstance(inputs, dict) and "properties" in inputs:
            if isinstance(inputs["properties"], dict):
                for input_name, input_schema in inputs["properties"].items():
                    if isinstance(input_schema, dict) and "$ref" in input_schema:
                        parameters[input_name] = {
                            "type": "string",
                            "description": f"Referenced input: {input_schema['$ref']}",
                        }
                        continue

                    if not isinstance(input_schema, dict):
                        parameters[input_name] = {
                            "type": "string",
                            "description": f"Input parameter {input_name}",
                        }
                        continue

                    param_schema = input_schema.copy()
                    if "description" not in param_schema:
                        param_schema["description"] = f"Input parameter {input_name}"
                    parameters[input_name] = param_schema

        return parameters

    def _extract_required_parameters(self, definition: dict[str, Any]) -> list[str]:
        """Extract required parameters from a workflow definition.

        Args:
            definition: Arazzo workflow specification.

        Returns:
            List of required parameter names.
        """
        required = []

        if "inputs" in definition and isinstance(definition["inputs"], dict):
            inputs_schema = definition["inputs"]
            if "required" in inputs_schema and isinstance(inputs_schema["required"], list):
                required = inputs_schema["required"]

        return required

    def _extract_operation_parameters(
        self, operation: dict[str, Any]
    ) -> tuple[dict[str, dict[str, Any]], list[str]]:
        """Extract parameters and required fields from an operation definition.

        Handles both direct properties and nested 'body' properties.

        Args:
            operation: Jentic operation specification.

        Returns:
            Tuple containing (parameters dictionary, required list).
        """
        parameters = {}
        required = []
        all_required = set()

        if "inputs" in operation and isinstance(operation["inputs"], dict):
            inputs_schema = operation["inputs"]

            if isinstance(inputs_schema.get("properties"), dict):
                for param_name, param_schema in inputs_schema["properties"].items():
                    if not isinstance(param_schema, dict):
                        parameters[param_name] = {
                            "type": "string",
                            "description": f"Parameter {param_name}",
                        }
                        continue

                    if param_name == "body" and isinstance(param_schema.get("properties"), dict):
                        body_schema = param_schema
                        body_properties = body_schema["properties"]
                        body_required = set(body_schema.get("required", []))

                        for body_param_name, body_param_schema in body_properties.items():
                            if isinstance(body_param_schema, dict):
                                param_copy = body_param_schema.copy()
                                if "description" not in param_copy:
                                    param_copy["description"] = f"Body parameter {body_param_name}"
                                parameters[body_param_name] = param_copy
                                if body_param_name in body_required:
                                    all_required.add(body_param_name)
                            else:
                                parameters[body_param_name] = {
                                    "type": "string",
                                    "description": f"Body parameter {body_param_name}",
                                }
                                if body_param_name in body_required:
                                    all_required.add(body_param_name)
                    else:
                        param_copy = param_schema.copy()
                        if "description" not in param_copy:
                            param_copy["description"] = f"Parameter {param_name}"
                        parameters[param_name] = param_copy
                        if param_schema.get("required") is True:
                            all_required.add(param_name)

            if isinstance(inputs_schema.get("required"), list):
                all_required.update(inputs_schema["required"])

        required = sorted(list(all_required))
        return parameters, required

    def _is_valid_parameter_name(self, name: str) -> bool:
        """Check if a parameter name is valid across LLM platforms.
        
        A valid parameter name must:
        1. Match the pattern [a-zA-Z0-9_-]+ (alphanumeric, underscore, hyphen)
        2. Be between 1 and 64 characters long
        3. Not be empty
        
        Args:
            name: The parameter name to validate
            
        Returns:
            True if the name is valid, False otherwise
        """
        return bool(name and self._valid_param_name_pattern.match(name))
        
    def _sanitize_vendor_name(self, vendor: str) -> str:
        """Sanitize a vendor name (from api_name) to be used in tool names.
        
        This ensures that vendor names from domains like 'discord.com' are properly
        sanitized to work in tool/function names for LLM platforms.
        
        Args:
            vendor: The vendor name to sanitize (typically from api_name)
            
        Returns:
            A sanitized vendor name safe to use in tool names, or empty string if invalid
        """
        # Replace invalid chars with hyphens (more readable than underscores for vendor names)
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "-", vendor)
        # Collapse consecutive hyphens
        sanitized = re.sub(r"-+", "-", sanitized)
        # Trim hyphens
        sanitized = sanitized.strip("-")
        # Return empty string for invalid vendor names
        return sanitized
    
    def _sanitize_parameter_name(self, name: str) -> str | None:
        """Return a sanitized parameter name that matches valid parameter patterns.

        The sanitization strategy is:
        1. Replace any disallowed character with an underscore.
        2. Collapse multiple consecutive underscores.
        3. Trim leading/trailing underscores.
        4. Truncate to 64 characters (compatible with all LLM platforms).
        5. Return None for empty strings or strings with only invalid chars so they can be excluded.

        Note: Tests may need to be updated as this function now returns None for invalid parameters
        instead of using 'param' as a fallback.
        """
        # Replace invalid chars with underscore
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        # Collapse consecutive underscores
        sanitized = re.sub(r"_+", "_", sanitized)
        # Trim underscores
        sanitized = sanitized.strip("_")
        # Return None for empty strings or strings with only invalid chars
        if not sanitized:
            return None
        # Truncate to 64 chars (for LLM platform compatibility)
        if len(sanitized) > 64:
            sanitized = sanitized[:64]
        return sanitized

    def _sanitize_parameters(
        self,
        tool_name: str,
        parameters: dict[str, dict[str, Any]],
        required: list[str],
    ) -> tuple[dict[str, dict[str, Any]], list[str]]:
        """Sanitize parameter keys to ensure consistency across LLM providers.
        
        Args:
            tool_name: Name of the tool being sanitized
            parameters: Dictionary of parameter schemas
            required: List of required parameter names
            
        Returns:
            Tuple of (sanitized_parameters, sanitized_required)
        """
        sanitized_parameters: dict[str, dict[str, Any]] = {}
        mapping: dict[str, str] = {}
        
        # Apply consistent sanitization to all parameters
        used_names = set()
        required_param_name = None
        
        # Process all parameters
        for original_name, schema in parameters.items():
            if self._is_valid_parameter_name(original_name):
                # Name is already valid, use as-is
                sanitized_name = original_name
            else:
                # Try to sanitize the name
                sanitized_name = self._sanitize_parameter_name(original_name)
                # If sanitization returns None (completely invalid), skip this parameter
                if sanitized_name is None:
                    logger.debug(f"Excluding parameter with invalid name: '{original_name}'")
                    continue
            
            # Handle collisions by adding a suffix
            if sanitized_name in used_names:
                base_name = sanitized_name
                counter = 1
                while sanitized_name in used_names:
                    sanitized_name = f"{base_name}_{counter}"
                    counter += 1
                    
                    # Make sure we stay within the 64-char limit
                    if len(sanitized_name) > 64:
                        # If we're going to exceed, truncate the base name to make room for the suffix
                        suffix = f"_{counter}"
                        base_name = base_name[:64 - len(suffix)]
                        sanitized_name = f"{base_name}{suffix}"
            
            used_names.add(sanitized_name)
            sanitized_parameters[sanitized_name] = schema
            if sanitized_name != original_name:
                mapping[sanitized_name] = original_name
        
        # Build a map from original name to sanitized name
        original_to_sanitized = {}
        for sanitized_name, original_name in mapping.items():
            original_to_sanitized[original_name] = sanitized_name
            
        # Update required list using the name mapping
        # Only include valid parameter names or ones we've successfully sanitized
        sanitized_required = []
        for req_name in required:
            if self._is_valid_parameter_name(req_name) and req_name in parameters:
                # If the name is already valid and exists in the parameters, use it directly
                sanitized_required.append(req_name)
            elif req_name in original_to_sanitized and original_to_sanitized[req_name] in sanitized_parameters:
                # If we have a mapping for this required parameter and it's in our sanitized parameters, use it
                sanitized_required.append(original_to_sanitized[req_name])
            else:
                # If the parameter can't be sanitized or isn't in the sanitized parameters, we exclude it
                sanitized_name = None
                if not self._is_valid_parameter_name(req_name):
                    sanitized_name = self._sanitize_parameter_name(req_name)
                else:
                    sanitized_name = req_name
                    
                # Skip invalid parameters
                if sanitized_name is None or sanitized_name not in sanitized_parameters:
                    logger.debug(f"Excluding required parameter '{req_name}' as it has an invalid name or was not found in parameters")
                    continue
                    
                # Otherwise, add it to the required list
                sanitized_required.append(sanitized_name)
            
        # Persist mapping if any substitutions occurred
        if mapping:
            self._parameter_mappings[tool_name] = mapping
            
        return sanitized_parameters, sanitized_required

    def restore_input_parameter_names(self, tool_name: str, inputs: dict[str, Any]) -> dict[str, Any]:
        """Restore original parameter names for execution using stored mappings.

        Args:
            tool_name: Name of the tool whose inputs are being restored.
            inputs: Dictionary of inputs received from the LLM (potentially sanitized).

        Returns:
            Dictionary with keys converted back to their original names expected by the
            workflow/operation runtime.
        """
        if not inputs:
            return inputs
        mapping = self._parameter_mappings.get(tool_name, {})
        if not mapping:
            return inputs  # No sanitization performed for this tool

        restored: dict[str, Any] = {}
        for key, value in inputs.items():
            original_key = mapping.get(key, key)
            restored[original_key] = value
        return restored

    def get_tool_type(self, tool_name: str) -> Literal["workflow", "operation", "unknown"]:
        """Determine if a tool name corresponds to a workflow or an operation."""
        if tool_name in self._workflow_definitions:
            return "workflow"
        elif tool_name in self._operation_name_to_uuid:
            return "operation"
        else:
            logger.warning(f"Tool name '{tool_name}' not found in workflows or mapped operations.")
            return "unknown"

    def get_operation_uuid(self, operation_tool_name: str) -> str | None:
        """Get the UUID for a given operation tool name."""
        return self._operation_name_to_uuid.get(operation_tool_name)

    def get_workflow_uuid(self, tool_name: str) -> str | None:
        """Get the UUID for a given workflow tool name."""
        return self._workflow_name_to_uuid.get(tool_name)


# Factory function
def create_llm_tool_manager() -> LLMToolSpecManager:
    """Create an instance of the LLM tool specification manager.

    Returns:
        LLM tool specification manager.
    """
    return LLMToolSpecManager()
