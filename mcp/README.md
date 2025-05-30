# Jentic MCP Plugin
[![smithery badge](https://smithery.ai/badge/@jentic/jentic-tools)](https://smithery.ai/server/@jentic/jentic-tools)


## Why Use Jentic MCP?

Jentic MCP empowers developers to discover and integrate external APIs and workflows rapidly—without the need to write or maintain any API-specific code. By leveraging the MCP protocol and Jentic’s agentic runtime, developers can:

- Search for APIs and workflows by capability, not by vendor or implementation.
- Instantly generate integration code samples that are agnostic to specific API details.
- Avoid boilerplate and reduce maintenance by relying on standardized, declarative integration.
- Focus on building features, while Jentic MCP handles the complexity of API interaction.

## API Tools

The Jentic MCP Plugin provides the following tools:

1. `search_apis`: Search for APIs in the Jentic directory that match specific functionality needs
2. `load_execution_info`: Retrieve detailed specifications for APIs and operations from the Jentic directory. **This will include auth information you may need to provide in your `mcpServers.jentic.env` configuration.**
3. `execute`: Execute a specific API or workflow operation.

## Getting Started

The recommended method is to run the server directly from the GitHub repository using `uvx`. 
You will need to install `uv` first using:

`brew install uv` or `pip install uv`

### Get Your Jentic UUID

To use the Jentic SDK, you must first obtain a Jentic UUID. The easiest way is using the Jentic CLI. You can _optionally_ include an email address for higher rate limits and for early access to new features.

```sh
pip install jentic
jentic register --email '<your_email>'
```

This will print your UUID and an export command to set it in your environment:

```sh
export JENTIC_UUID=<your-jentic-uuid>
```

Set the Jentic UUID in your MCP client configuration as shown below.

The location of the configuration file depends on the client you are using and your OS. Some common examples:

- **Windsurf**: `~/.codeium/windsurf/mcp_config.json`
- **Claude Desktop**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Claude Code**: `~/.claude.json`
- **Cursor**: `~/cursor/.mcp.json`


For other clients, check your client's documentation for how to add MCP servers.

```json
{
    "mcpServers": {
        "jentic": {
            "command": "uvx",
            "args": [
                "--from",
                "git+https://github.com/jentic/jentic-tools.git@main#subdirectory=mcp",
                "mcp"
            ],
            "env": {
                "JENTIC_UUID": "<your-jentic-uuid>"
            }
        }
    }
}
```

__Note:__ After saving the configuration file, you may need to restart the client application (Windsurf, Claude Desktop) for the changes to take effect.

### MCP Tool Use

Once the MCP server is running, you can easily use the MCP tools in your LLM agent to discover and execute APIs and workflows.

1. `search_apis`: Search for APIs in the Jentic directory that match specific functionality needs
2. `load_execution_info`: Retrieve detailed specifications for APIs and operations from the Jentic directory. **This will include auth information you may need to provide in your `mcpServers.jentic.env` configuration.**
3. `execute`: Execute a specific API or workflow operation.

### Environment Variables

When you are using an API that requires authentication, the `load_execution_info` tool will describe the required environment variables. You environment variables via the command line in Windsurf, although in some clients like Claude Desktop, you'll need to add them to your MCP config:

```json
{
    "mcpServers": {
        "jentic": {
            "command": "uvx",
            "args": [
                "--from",
                "git+https://github.com/jentic/jentic-tools.git@main#subdirectory=mcp",
                "mcp"
            ],
            "env": {
                "DISCORD_BOTTOKEN=": "YOUR BOT TOKEN"
            }
        }
    }
}
```

**Alternative (Using Local Path for Development):**

Use this if you are actively developing the MCP plugin locally. Replace `/path/to/your/project/mcp` with the absolute path to your project directory.

```json
{
    "mcpServers": {
        "jentic": { 
            "command": "uvx",
            "args": [
                "--from",
                "/path/to/your/project/mcp",
                "mcp" 
            ]
        }
    }
}
```

_Optional:_ Add a `JENTIC_API_URL` environment variable to your `mcp_config.json` file to point to a specific Jentic API (works with both methods):

```json
{
    "mcpServers": {
        "jentic": {
            "command": "uvx",
            "args": [
                "--from",
                "git+https://github.com/jentic/jentic-tools.git@main#subdirectory=mcp",
                "mcp"
            ],
            "env": {
                "JENTIC_API_URL": "https://your-jentic-api.url/"
            }
        }
    }
}
```

Once configured, restart Windsurf, and the Jentic MCP tools will be available.

You can tail the logs generated by the locally running MCP server by running:

```bash
tail /path/to/mcp/jentic_ark2_mcp.log
```

## Installation

### Installing via Smithery

To install Jentic Plugin for Claude Desktop automatically via [Smithery](https://smithery.ai/server/@jentic/jentic-tools):

```bash
npx -y @smithery/cli install @jentic/jentic-tools --client claude
```

### Manual Installation
Ensure you have `pdm` installed (`pipx install pdm`).

To install the package and its dependencies for development:

```bash
# From the project root directory (e.g., /Users/kc/c/sdk/mcp)
pdm install -G dev
```

## Running the MCP Server

The Jentic MCP plugin is designed to be run using `uvx`, which handles environment setup and execution.

### Default Mode (Stdio)

Run the MCP plugin directly using `uvx`, specifying the project directory *as the source* using `--from` and the `mcp` script:

**From Local Path (Development):**

```bash
# Use --from with the project directory and specify the 'mcp' script
uvx --from /path/to/your/project/mcp mcp

# Or, if running from within the project directory:
uvx --from . mcp
```

**From Remote Repository (Recommended for general use):**

```bash
uvx --from \
  git+https://github.com/jentic/jentic-tools.git@main#subdirectory=mcp \
  mcp
```

This automatically uses the default `serve --transport stdio` command defined in the `mcp` script's callback.

### HTTP Mode

To run the server in HTTP mode (e.g., for testing with `claude-cli`):

**From Local Path (Development):**

```bash
# Default HTTP (port 8010)
uvx --from /path/to/your/project/mcp mcp serve --transport http

# With custom port
uvx --from /path/to/your/project/mcp mcp serve --transport http --port 8080

# With custom host
uvx --from /path/to/your/project/mcp mcp serve --transport http --host 0.0.0.0 --port 8080
```

**From Remote Repository (Recommended):**

```bash
uvx --from \
  git+https://github.com/jentic/jentic-tools.git@main#subdirectory=mcp \
  mcp serve --transport http --port 8080
```

### Running from a Remote Git Repository

You can also run the MCP server directly from a Git repository URL without cloning it locally using `uvx --from`:

```bash
# Example from a specific branch and subdirectory
uvx --from \
  git+https://github.com/jentic/jentic-tools.git@main#subdirectory=mcp

# Explanation:
# - git+https://... .git : The repository URL
# - @main   : The branch, tag, or commit hash
# - #subdirectory=mcp     : The folder within the repo containing the pyproject.toml
```

This command fetches the specified version, installs its dependencies in a temporary environment, and runs the default command (which is `serve --transport stdio`).

You can add other arguments like `--log-level DEBUG` or `--mock` after the URL fragment:

```bash
uvx --from \
  git+https://github.com/jentic/jentic-tools.git@main#subdirectory=mcp \
  --log-level DEBUG --mock
```

To run in HTTP mode from a remote source:

```bash
uvx --from \
  git+https://github.com/jentic/jentic-tools.git@main#subdirectory=mcp \
  serve --transport http --port 8080
```

### Other Options

#### Logging

```bash
# Set logging level (applies to default stdio or explicit serve)
uvx --from /path/to/your/project/mcp mcp --log-level DEBUG
uvx --from /path/to/your/project/mcp mcp serve --transport http --log-level DEBUG

# Log to file (applies to default stdio or explicit serve)
uvx --from /path/to/your/project/mcp mcp --log-file jentic_mcp.log
uvx --from /path/to/your/project/mcp mcp serve --transport http --log-file jentic_mcp.log
```

#### Mock Mode

Enable mock mode for development without connecting to the actual Jentic API Knowledge Hub:

```bash
# Mock mode with default stdio
uvx --from /path/to/your/project/mcp mcp --mock

# Mock mode with explicit HTTP
uvx --from /path/to/your/project/mcp mcp serve --transport http --mock
```

#### Environment Variables

Provide environment variables using a `.env` file:

```bash
# Env file with default stdio
uvx --from /path/to/your/project/mcp mcp --env-file .env

# Env file with explicit HTTP
uvx --from /path/to/your/project/mcp mcp serve --transport http --env-file .env
```

### Using with Claude

The MCP plugin can be used with Claude or other LLMs that support the MCP specification:

**Run from Remote Repository (Recommended):**

```
# Run the server in HTTP mode first
uvx --from \
  git+https://github.com/jentic/jentic-tools.git@main#subdirectory=mcp \
  mcp serve --transport http --port 8000

# Then connect claude-cli
claude-cli --mcp http://localhost:8000
```

**Run from Local Path (Development):**

```
# Run the server in HTTP mode first
uvx --from /path/to/your/project/mcp mcp serve --transport http --port 8000

# Then connect claude-cli
claude-cli --mcp http://localhost:8000
```

## Development

See [CLAUDE.md](CLAUDE.md) for detailed development instructions.

### Package Structure

- `src/mcp/`: Main MCP package
  - `transport/`: Transport implementations (HTTP, stdio)
  - `mock/`: Mock data providers for development
  - `tools.py`: Tool definitions
  - `handlers.py`: Request handlers
  - `main.py`: CLI entry points
  - `adapters/`: Adapter implementations
  - `core/`: Core functionality
- `tests/`: Test suite

### Testing

```bash
# Ensure dev dependencies are installed: pdm install -G dev
pdm run test
```

### Linting & Formatting

Uses `ruff`, `black`, `isort`, `mypy` via pdm scripts.

```bash
# Run all linters/formatters
pdm run lint

# Run only ruff
pdm run linter
```

## License

Proprietary - Jentic
