import asyncio

from mcp_client import MCPClient


async def main():

    client = MCPClient()

    await client.connect()

    print("\n==============================")
    print("AVAILABLE MCP TOOLS")
    print("==============================")

    tools = await client.list_tools()

    for name, info in tools.items():

        print(
            f"{name:15} "
            f"→ {info['server']}"
        )

    print("\n==============================")

    await client.close()


if __name__ == "__main__":

    asyncio.run(main())