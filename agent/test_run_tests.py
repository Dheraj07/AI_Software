import asyncio

from agent.mcp_client import MCPClient


async def main():

    client = MCPClient()

    try:
        print("\nConnecting to MCP servers...")
        await client.connect()

        tools = client.get_langchain_tools()

        run_tests_tool = next(
            tool for tool in tools
            if tool.name == "run_tests"
        )

        print("\nCalling LangChain run_tests tool...")

        result = await run_tests_tool.ainvoke({})

        print("\n==============================")
        print("LANGCHAIN TOOL RESULT")
        print("==============================")
        print(result)

    finally:
        await client.exit_stack.aclose()


if __name__ == "__main__":
    asyncio.run(main())