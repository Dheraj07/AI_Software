import asyncio

from langchain_core.tools import tool

from mcp_client import MCPClient


async def main():

    print("1. Connecting...")

    mcp = MCPClient()

    await mcp.connect()

    print("2. Connected")
    tools = await mcp.list_tools()

    print(
        "Available tools:",
        list(tools.keys())
)

    @tool
    async def read_file(file_path: str) -> str:
        """Read a text file from the workspace."""

        print(
            f"Calling MCP read_file: {file_path}"
        )

        result = await mcp.call_tool(
            "read_file",
            {
                "file_path": file_path
            }
        )

        if result.structured_content:
            return str(result.structured_content)

        return str(result.content)

    print("3. Tool created")

    print("4. Calling tool...")

    result = await read_file.ainvoke(
        {
            # "file_path": "workspace/calculator/utils.py"
            "file_path": "/Users/dherajmajeti/Documents/ai-software-team/workspace/calculator/tests/test_utils.py"
        }
    )

    print("5. Result:")

    print(result)

    await mcp.close()


if __name__ == "__main__":
    asyncio.run(main())