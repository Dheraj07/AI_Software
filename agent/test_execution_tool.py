import asyncio

from mcp_client import MCPClient


async def main():

    print("1. Connecting...")

    mcp = MCPClient()

    await mcp.connect()

    print("2. Connected")

    print("3. Calling run_tests...")

    result = await mcp.call_tool(
        "run_tests",
        {}
    )

    print("4. Result:")

    print(result)

    await mcp.close()


if __name__ == "__main__":
    asyncio.run(main())