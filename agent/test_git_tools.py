import asyncio

from mcp_client import MCPClient


async def main():

    print("1. Connecting...")

    mcp = MCPClient()

    await mcp.connect()

    print("2. Connected")

    # -----------------------------------------
    # STATUS
    # -----------------------------------------

    print("\n3. Git status...")

    result = await mcp.call_tool(
        "git_status",
        {}
    )

    print(result)

    # -----------------------------------------
    # DIFF
    # -----------------------------------------

    print("\n4. Git diff...")

    result = await mcp.call_tool(
        "git_diff",
        {}
    )

    print(result)

    # -----------------------------------------
    # ADD
    # -----------------------------------------

    print("\n5. Git add...")

    result = await mcp.call_tool(
        "git_add",
        {
            "files": [
                "mcp_server/git_server.py"
            ]
        }
    )

    print(result)

    # -----------------------------------------
    # STATUS AFTER ADD
    # -----------------------------------------

    print("\n6. Git status after add...")

    result = await mcp.call_tool(
        "git_status",
        {}
    )

    print(result)

    # -----------------------------------------
    # COMMIT
    # -----------------------------------------

    print("\n7. Git commit...")

    result = await mcp.call_tool(
        "git_commit",
        {
            "message": "Add Git MCP server"
        }
    )

    print(result)

    # -----------------------------------------
    # FINAL STATUS
    # -----------------------------------------

    print("\n8. Final Git status...")

    result = await mcp.call_tool(
        "git_status",
        {}
    )

    print(result)

    await mcp.close()


if __name__ == "__main__":
    asyncio.run(main())