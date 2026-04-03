# Global Agent Directives

## Core Protocol
1.  **Debug Verification:** Before implementing any solution, you MUST explicitly debug and verify the root cause of the problem. Do not attempt to solve unverified issues. Debug through prints, shape and label checks.

2.  **Strict Tool Adherence:** You are required to use specific MCP tools for their assigned domains. Native fallbacks are prohibited when these tools are applicable.

## Tool Mandates

### 1. File Operations: `serena`
**REQUIRED:** Use the `serena` MCP tool for all file system interactions.
* **Navigation:** Listing directories, finding files.
* **Reading:** Retrieving file content.
* **Editing:** Applying changes to code or configuration.

### 2. Knowledge Retrieval: `docs-mcp-server`
**REQUIRED:** Use the `docs-mcp-server` MCP tool for external information.
* **Library Lookups:** Querying syntax, functions, and modules.
* **Documentation:** Fetching official guides for specific technologies.
* **Validation:** Verifying API usage against current documentation before coding.
