import asyncio

from mcp_client import MCPClient


async def main():

    client = MCPClient()

    await client.connect()

    await client.list_tools()

    tools = client.get_langchain_tools()

    print("\n==============================")
    print("LANGCHAIN TOOLS")
    print("==============================")

    for tool in tools:

        print(
            f"\nName: {tool.name}"
        )

        print(
            f"Description: "
            f"{tool.description}"
        )

    print(
        "\nTotal tools:",
        len(tools)
    )

    await client.close()


if __name__ == "__main__":

    asyncio.run(main())