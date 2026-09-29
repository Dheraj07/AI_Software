import asyncio

from langchain_ollama import ChatOllama

from mcp_client import MCPClient
from mcp_tools import make_langchain_tool


async def main():

    print("Connecting to MCP...")

    mcp = MCPClient()

    await mcp.connect()

    mcp_tools = await mcp.list_tools()

    print("\nMCP tools:")

    for tool in mcp_tools:
        print(
            f"- {tool.name}: "
            f"{tool.description}"
        )

    # --------------------------------------------------
    # Convert only read_file for this test
    # --------------------------------------------------

    read_file_mcp = next(
        tool
        for tool in mcp_tools
        if tool.name == "read_file"
    )

    read_file = make_langchain_tool(
        mcp,
        read_file_mcp
    )

    print("\nCreated LangChain tool:")
    print(read_file.name)

    # --------------------------------------------------
    # Create Qwen
    # --------------------------------------------------

    llm = ChatOllama(
        model="qwen2.5-coder:14b",
        temperature=0,
        base_url="http://localhost:11434",
    )

    print("\nBinding read_file to Qwen...")

    llm_with_tools = llm.bind_tools(
        [read_file]
    )

    print("\nAsking Qwen to use read_file...")

    response = await asyncio.wait_for(
        llm_with_tools.ainvoke(
            """
Use the read_file tool to read:

calculator/utils.py

You MUST use the tool.
Do not answer from your own knowledge.
"""
        ),
        timeout=120,
    )

    print("\n========== RESPONSE ==========")

    print(response)

    print("\n========== TOOL CALLS ==========")

    print(response.tool_calls)

    await mcp.close()


if __name__ == "__main__":
    asyncio.run(main())