import asyncio

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage

from mcp_client import MCPClient


async def main():

    print("1. Connecting to MCP...")

    mcp = MCPClient()

    await mcp.connect()

    print("2. MCP connected")

    # ==================================================
    # MCP-BACKED TOOLS
    # ==================================================

    @tool
    async def list_files(directory: str = ".") -> str:
        """List files and directories inside the workspace."""

        result = await mcp.call_tool(
            "list_files",
            {"directory": directory}
        )

        if result.is_error:
            return f"ERROR: {result.content}"

        return str(
            result.structured_content
            or result.content
        )

    @tool
    async def read_file(file_path: str) -> str:
        """Read a text file inside the workspace."""

        print(
            f"\n📖 MCP read_file → {file_path}"
        )

        result = await mcp.call_tool(
            "read_file",
            {"file_path": file_path}
        )

        if result.is_error:
            return f"ERROR: {result.content}"

        return str(
            result.structured_content
            or result.content
        )

    @tool
    async def write_file(
        file_path: str,
        content: str
    ) -> str:
        """Write or replace a text file."""

        print(
            f"\n✍️ MCP write_file → {file_path}"
        )

        result = await mcp.call_tool(
            "write_file",
            {
                "file_path": file_path,
                "content": content
            }
        )

        if result.is_error:
            return f"ERROR: {result.content}"

        return str(
            result.structured_content
            or result.content
        )

    tools = [
        list_files,
        read_file,
        write_file
    ]

    tool_map = {
        t.name: t
        for t in tools
    }

    print("\n3. Tools registered:")

    for t in tools:
        print(f"   - {t.name}")

    # ==================================================
    # QWEN
    # ==================================================

    print("\n4. Creating Qwen3...")

    llm = ChatOllama(
        model="qwen3:8b",
        temperature=0,
        base_url="http://localhost:11434"
    )

    llm_with_tools = llm.bind_tools(tools)

    print("5. Qwen3 ready")

    # ==================================================
    # TASK
    # ==================================================

    messages = [
        (
            "system",
            """
You are an autonomous software engineering agent.

You have access to an MCP filesystem.

Your tools are:

- list_files
- read_file
- write_file

Follow these rules:

1. Read a file before modifying it.
2. Never invent existing code.
3. Preserve existing functionality.
4. Make only the requested change.
5. After modifying a file, read it again to verify.
6. Continue until the user's task is completely finished.
"""
        ),
        (
            "human",
            """
Modify calculator/utils.py.

Add a function:

def multiply(a, b):
    return a * b

Do not remove or modify the existing functions.

After making the change, verify the final file.
"""
        )
    ]

    # ==================================================
    # AGENT LOOP
    # ==================================================

    for iteration in range(8):

        print(
            f"\n{'=' * 60}"
        )

        print(
            f"🤖 AGENT ITERATION {iteration + 1}"
        )

        print(
            f"{'=' * 60}"
        )

        response = await asyncio.wait_for(
            llm_with_tools.ainvoke(messages),
            timeout=120
        )

        print("\nQWEN RESPONSE")

        if response.content:
            print(response.content)

        print(
            "\nTOOL CALLS:"
        )

        print(
            response.tool_calls
        )

        # Add Qwen response to history
        messages.append(response)

        # ----------------------------------------------
        # No tools = task finished
        # ----------------------------------------------

        if not response.tool_calls:

            print(
                "\n✅ Agent finished."
            )

            break

        # ----------------------------------------------
        # Execute tools
        # ----------------------------------------------

        for call in response.tool_calls:

            name = call["name"]
            args = call["args"]

            print(
                f"\n🔧 Executing: {name}"
            )

            print(
                f"Arguments: {args}"
            )

            selected_tool = tool_map.get(name)

            if selected_tool is None:

                result = (
                    f"ERROR: Unknown tool {name}"
                )

            else:

                try:

                    result = await selected_tool.ainvoke(
                        args
                    )

                except Exception as e:

                    result = (
                        f"ERROR: {type(e).__name__}: {e}"
                    )

            print(
                "\n📦 RESULT:"
            )

            print(result)

            # Give result back to Qwen

            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=call["id"]
                )
            )

    await mcp.close()

    print(
        "\nMCP connection closed."
    )


if __name__ == "__main__":
    asyncio.run(main())