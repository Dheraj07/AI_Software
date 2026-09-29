import asyncio

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict
from langchain_core.tools import StructuredTool
from mcp_client import MCPClient


# ============================================================
# STATE
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ============================================================
# MCP + TOOLS
# ============================================================

mcp = MCPClient()


# @tool
# async def list_files(directory: str = ".") -> str:
#     """List files and directories inside the workspace."""

#     result = await mcp.call_tool(
#         "list_files",
#         {
#             "directory": directory
#         }
#     )

#     if result.is_error:
#         return f"ERROR: {result.content}"

#     return str(
#         result.structured_content
#         or result.content
#     )


# @tool
# async def read_file(file_path: str) -> str:
#     """Read a text file inside the workspace."""

#     print(
#         f"\n📖 MCP read_file → {file_path}"
#     )

#     result = await mcp.call_tool(
#         "read_file",
#         {
#             "file_path": file_path
#         }
#     )

#     if result.is_error:
#         return f"ERROR: {result.content}"

#     return str(
#         result.structured_content
#         or result.content
#     )


# @tool
# async def write_file(
#     file_path: str,
#     content: str
# ) -> str:
#     """Write or replace a text file."""

#     print(
#         f"\n✍️ MCP write_file → {file_path}"
#     )

#     result = await mcp.call_tool(
#         "write_file",
#         {
#             "file_path": file_path,
#             "content": content
#         }
#     )

#     if result.is_error:
#         return f"ERROR: {result.content}"

#     return str(
#         result.structured_content
#         or result.content
#     )

# @tool
# async def run_python(code: str) -> str:
#     """Execute Python code inside the workspace."""

#     print("\n🐍 MCP run_python")

#     result = await mcp.call_tool(
#         "run_python",
#         {
#             "code": code
#         }
#     )

#     if result.is_error:
#         return f"ERROR: {result.content}"

#     return str(
#         result.structured_content
#         or result.content
#     )

# @tool
# async def run_tests() -> str:
#     """Run the project's pytest test suite."""

#     print("\n🧪 MCP run_tests")

#     result = await mcp.call_tool(
#         "run_tests",
#         {}
#     )

#     if result.is_error:
#         return f"ERROR: {result.content}"

#     return str(
#         result.structured_content
#         or result.content
#     )



# tools = [
#     list_files,
#     read_file,
#     write_file,
#     run_python,
#     run_tests
# ]


# tool_map = {
#     tool.name: tool
#     for tool in tools
# }


# ============================================================
# LLM
# ============================================================

llm = ChatOllama(
    model="qwen3:8b",
    temperature=0,
    base_url="http://localhost:11434",
)

# llm_with_tools = llm.bind_tools(tools)


# ============================================================
# AGENT NODE
# ============================================================

async def agent_node(state: AgentState):

    print("\n" + "=" * 60)
    print("🤖 AGENT NODE")
    print("=" * 60)

    response = await llm_with_tools.ainvoke(
        state["messages"]
    )

    if response.tool_calls:

        print("\nQwen requested:")

        for call in response.tool_calls:

            print(
                f"  🔧 {call['name']}"
            )

            print(
                f"     {call['args']}"
            )

    else:

        print("\nQwen produced final answer.")

    return {
        "messages": [response]
    }


# ============================================================
# TOOL NODE
# ============================================================

async def tool_node(state: AgentState):

    print("\n" + "=" * 60)
    print("🛠️ MCP TOOL NODE")
    print("=" * 60)

    last_message = state["messages"][-1]

    results = []

    for call in last_message.tool_calls:

        tool_name = call["name"]
        arguments = call["args"]

        print(
            f"\nExecuting: {tool_name}"
        )

        print(
            f"Arguments: {arguments}"
        )

        selected_tool = tool_map.get(
            tool_name
        )

        if selected_tool is None:

            result = (
                f"ERROR: Unknown tool: "
                f"{tool_name}"
            )

        else:

            try:

                result = await selected_tool.ainvoke(
                    arguments
                )

            except Exception as e:

                result = (
                    f"ERROR: "
                    f"{type(e).__name__}: {e}"
                )

        print("\nResult:")

        print(result)

        results.append(
            ToolMessage(
                content=str(result),
                tool_call_id=call["id"],
            )
        )

    return {
        "messages": results
    }


# ============================================================
# ROUTER
# ============================================================

def should_continue(state: AgentState):

    last_message = state["messages"][-1]

    if last_message.tool_calls:

        return "tools"

    return END


# ============================================================
# BUILD GRAPH
# ============================================================

def build_graph(tools):

    llm_with_tools = llm.bind_tools(tools)

    tool_map = {
        tool.name: tool
        for tool in tools
    }

    graph = StateGraph(
        AgentState
    )

    async def dynamic_agent_node(state: AgentState):

        print("\n" + "=" * 60)
        print("🤖 AGENT NODE")
        print("=" * 60)

        response = await llm_with_tools.ainvoke(
            state["messages"]
        )

        if response.tool_calls:

            print("\nQwen requested:")

            for call in response.tool_calls:

                print(
                    f"  🔧 {call['name']}"
                )

                print(
                    f"     {call['args']}"
                )

        else:

            print("\nQwen produced final answer.")

        return {
            "messages": [response]
        }

    async def dynamic_tool_node(state: AgentState):

        print("\n" + "=" * 60)
        print("🛠️ MCP TOOL NODE")
        print("=" * 60)

        last_message = state["messages"][-1]

        results = []

        for call in last_message.tool_calls:

            tool_name = call["name"]
            arguments = call["args"]

            print(f"\nExecuting: {tool_name}")
            print(f"Arguments: {arguments}")

            selected_tool = tool_map.get(tool_name)

            if selected_tool is None:

                result = (
                    f"ERROR: Unknown tool: "
                    f"{tool_name}"
                )

            else:

                try:

                    result = await selected_tool.ainvoke(
                        arguments
                    )

                except Exception as e:

                    result = (
                        f"ERROR: "
                        f"{type(e).__name__}: {e}"
                    )

            print("\nResult:")
            print(result)

            results.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=call["id"],
                )
            )

        return {
            "messages": results
        }

    graph.add_node(
        "agent",
        dynamic_agent_node
    )

    graph.add_node(
        "tools",
        dynamic_tool_node
    )

    graph.add_edge(
        START,
        "agent"
    )

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END,
        }
    )

    graph.add_edge(
        "tools",
        "agent"
    )

    return graph.compile()


# ============================================================
# MAIN
# ============================================================

async def main():

    print("Connecting to MCP...")

    await mcp.connect()

    print("MCP connected.")
    await mcp.list_tools()

    tools = mcp.get_langchain_tools()

    print("\nDiscovered MCP tools:")

    for tool in tools:
        print(f"  - {tool.name}")

    graph = build_graph(tools)

    print("LangGraph compiled.")

    # --------------------------------------------------------
    # USER TASK
    # --------------------------------------------------------

    task = """
Inspect calculator/utils.py and make sure it is correct.

Requirements:

1. Inspect the complete file.
2. Fix any obvious bugs in the existing functions.
3. Add this function if it does not already exist:

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

4. Preserve all existing functionality.
5. Run the project's test suite using the run_tests tool.
6. If tests fail:
   - inspect the failure
   - fix the relevant code
   - run the tests again
7. Do not finish until the test suite passes.
8. Finally, read the modified file and summarize the changes.
"""

    initial_state = {
        "messages": [
            (
    "system",
    """
You are an autonomous software engineering agent.

You have access to an MCP filesystem.

Available tools:

- list_files
- read_file
- write_file

Follow this workflow for every coding task:

1. READ the complete target file before making any decision.

2. UNDERSTAND the existing code, including all existing
   functions and their implementations.

3. Check whether the user's requested change is already present.

4. Check the existing code for obvious errors that affect
   correctness, such as:
   - undefined variables
   - misspelled variable names
   - syntax errors
   - broken references
   - incorrect function implementations

5. If the requested change is missing, implement it.

6. If you discover an obvious existing bug while inspecting
   the file, report it to the user and fix it only when the
   user's task permits correcting existing code.

7. Never invent file contents.

8. Preserve existing functionality.

9. After EVERY write operation, read the file again.

10. Verify that:
    - the requested change exists
    - existing functions still exist
    - the resulting code is internally consistent

11. Do not declare the task complete merely because the
    requested function already exists.

12. Before finishing, review the entire modified file and
    explain what was changed or what existing issues were found.
"""
),
            (
                "human",
                task
            )
        ]
    }

    print("\nStarting LangGraph agent...")

    final_state = await graph.ainvoke(
        initial_state
    )

    print("\n" + "=" * 60)
    print("✅ FINAL RESULT")
    print("=" * 60)

    final_message = final_state["messages"][-1]

    print(
        final_message.content
    )

    await mcp.close()

    print(
        "\nMCP connection closed."
    )


if __name__ == "__main__":

    asyncio.run(main())