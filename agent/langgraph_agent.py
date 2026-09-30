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


@tool
async def list_files(directory: str = ".") -> str:
    """List files and directories inside the workspace."""

    result = await mcp.call_tool(
        "list_files",
        {
            "directory": directory
        }
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
        {
            "file_path": file_path
        }
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

@tool
async def run_python(code: str) -> str:
    """Execute Python code inside the workspace."""

    print("\n🐍 MCP run_python")

    result = await mcp.call_tool(
        "run_python",
        {
            "code": code
        }
    )

    if result.is_error:
        return f"ERROR: {result.content}"

    return str(
        result.structured_content
        or result.content
    )

@tool
async def run_tests() -> str:
    """Run the project's pytest test suite."""

    print("\n🧪 MCP run_tests")

    result = await mcp.call_tool(
        "run_tests",
        {}
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
    write_file,
    run_python,
    run_tests
]


tool_map = {
    tool.name: tool
    for tool in tools
}


# ============================================================
# LLM
# ============================================================

llm = ChatOllama(
    model="qwen3:8b",
    temperature=0,
    base_url="http://localhost:11434",
)

llm_with_tools = llm.bind_tools(tools)


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
    
Check the current Git repository status.

Use the git_status tool and report:
1. Current branch
2. Modified files
3. Untracked files
4. Whether the working tree is clean
"""

    initial_state = {
        "messages": [
            (
    "system",
    """
You are an autonomous software engineering agent.

You work inside a workspace using MCP tools.

Available capabilities:

Filesystem:
- list_files
- read_file
- write_file

Execution:
- run_python
- run_tests

Follow this workflow strictly.

============================================================
PHASE 1 — INSPECT
============================================================

1. Identify the target file.

2. Read the COMPLETE target file using read_file.

3. Understand all existing functions and their implementations.

4. Do not assume the file contents.

============================================================
PHASE 2 — ANALYZE
============================================================

5. Check whether the requested change already exists.

6. Inspect the existing implementation for obvious correctness
   problems such as:

   - syntax errors
   - misspelled variables
   - undefined variables
   - incorrect calculations
   - broken references
   - incorrect function behavior

7. Distinguish between:
   - requested changes
   - existing bugs
   - unrelated code

8. Preserve existing functionality.

============================================================
PHASE 3 — MODIFY
============================================================

9. If changes are required, use write_file.

10. Never invent existing code.

11. Make the smallest reasonable modification.

12. After EVERY write_file operation, immediately call
    read_file again.

13. Verify that:
    - the requested change exists
    - existing functions still exist
    - the code is internally consistent

============================================================
PHASE 4 — TEST
============================================================

14. ALWAYS run run_tests before declaring the task complete.

15. Do NOT say that tests would pass.

16. Actually execute run_tests.

============================================================
PHASE 5 — FAILURE LOOP
============================================================

17. If run_tests PASSES:

    - read the final modified file
    - verify the requested functionality
    - proceed to the final response

18. If run_tests FAILS:

    - inspect the COMPLETE test output
    - determine the likely cause
    - inspect the relevant source file if necessary
    - fix the problem using write_file
    - read the modified file again
    - run run_tests again

19. Repeat the fix → read → test cycle until the tests pass.

20. Never declare success while tests are failing.

============================================================
PHASE 6 — FINAL VERIFICATION
============================================================

21. Once tests pass, read the final target file.

22. Confirm that the requested change exists.

23. Confirm existing functionality was preserved.

24. Only then provide the final answer.

============================================================
IMPORTANT RULES
============================================================

- Never claim a test passed without actually running run_tests.
- Never stop after merely discovering that a requested function
  already exists.
- Never ignore a test failure.
- Never modify unrelated files unless required to fix the tests.
- Do not repeatedly make unnecessary changes.
- Prefer minimal, targeted fixes.

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