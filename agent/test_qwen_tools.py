import asyncio

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage

from mcp_client import MCPClient


async def main():

    print("1. Connecting to MCP...")

    mcp = MCPClient()

    await mcp.connect()

    print("2. Connected to MCP")

    @tool
    async def read_file(file_path: str) -> str:
        """
        Read a text file from the MCP workspace.
        """

        print(
            f"\n🔧 MCP read_file called: {file_path}"
        )

        result = await mcp.call_tool(
            "read_file",
            {
                "file_path": file_path
            }
        )

        if result.is_error:
            return f"ERROR: {result.content}"

        if result.structured_content:
            return str(result.structured_content)

        return str(result.content)

    print("3. Creating Qwen3...")

    llm = ChatOllama(
        model="qwen3:8b",
        temperature=0,
        base_url="http://localhost:11434",
    )

    print("4. Binding tool...")

    llm_with_tools = llm.bind_tools(
        [read_file]
    )

    # ------------------------------------------------
    # Conversation
    # ------------------------------------------------

    messages = [
        (
            "system",
            """
You are a software engineering assistant.

Use the available tools when you need information
from the workspace.

After receiving a tool result, answer the user.
"""
        ),
        (
            "human",
            """
Read calculator/utils.py and tell me what
functions are currently defined in the file.
"""
        ),
    ]

    # ------------------------------------------------
    # First Qwen call
    # ------------------------------------------------

    print("\n5. Calling Qwen3...")

    response = await llm_with_tools.ainvoke(
        messages
    )

    print("\n========== FIRST RESPONSE ==========")

    print(response)

    if not response.tool_calls:

        print(
            "\n⚠️ Qwen did not request a tool."
        )

        await mcp.close()

        return

    # ------------------------------------------------
    # Add Qwen response
    # ------------------------------------------------

    messages.append(response)

    # ------------------------------------------------
    # Execute tool calls
    # ------------------------------------------------

    for call in response.tool_calls:

        print(
            f"\nQwen requested: {call['name']}"
        )

        result = await read_file.ainvoke(
            call["args"]
        )

        print(
            "\n========== MCP RESULT =========="
        )

        print(result)

        # Add tool result to conversation

        messages.append(
            ToolMessage(
                content=result,
                tool_call_id=call["id"],
            )
        )

    # ------------------------------------------------
    # Second Qwen call
    # ------------------------------------------------

    print(
        "\n6. Sending MCP result back to Qwen..."
    )

    final_response = await llm_with_tools.ainvoke(
        messages
    )

    print(
        "\n========== FINAL RESPONSE =========="
    )

    print(
        final_response.content
    )

    await mcp.close()


if __name__ == "__main__":

    asyncio.run(main())