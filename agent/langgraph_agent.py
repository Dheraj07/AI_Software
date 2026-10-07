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
    test_passed: bool
    retry_count: int
    max_retries: int
    ran_tests: bool


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

def verify_node(state: AgentState):

    print("\n" + "=" * 60)
    print("🔍 VERIFY NODE")
    print("=" * 60)

    if state["ran_tests"]:
        print("✅ Tests have already been executed.")

        if state["test_passed"]:
            print("✅ Tests passed → END")
            return {}

        print("❌ Tests failed → AGENT")
        return {}

    print("⚠️ Tests have NOT been executed.")
    print("🧪 → TEST NODE")

    return {}
# ============================================================
# ROUTER
# ============================================================

def should_continue(state: AgentState):

    last_message = state["messages"][-1]

    if last_message.tool_calls:

        return "tools"
    # Do not allow the agent to finish before
    # the test suite has actually passed.
    # if not state["test_passed"]:
    #     print("\n⚠️ Agent attempted to finish before tests passed.")
    #     print("🧪 Forcing TEST NODE...")
    #     return "test"

    return "verify"

def test_router(state: AgentState):

    if state["test_passed"]:
        print("\n✅ Tests passed.")
        return "success"

    if state["retry_count"] < state["max_retries"]:
        print(
            f"\n❌ Tests failed. "
            f"Retry {state['retry_count']}/{state['max_retries']}"
            )
        return "fix"

    print(
            f"\n🛑 Maximum retries reached "
            f"({state['max_retries']})."
        )

    return "failed"


def verify_router(state: AgentState):

    if state["ran_tests"]:

        if state["test_passed"]:
            return "success"

        return "fix"

    return "test"

# async def test_node(state: AgentState):

#     print("\n" + "=" * 60)
#     print("🧪 TEST NODE")
#     print("=" * 60)

#     selected_tool = tool_map.get("run_tests")
#     if selected_tool is None:
#         print("❌ run_tests tool not found")

#         return {
#                 "messages": [
#                     ToolMessage(
#                         content="ERROR: run_tests tool not found",
#                         tool_call_id="test_node"
#                     )
#                 ],
#                 "test_passed": False,
#                 "retry_count": state["retry_count"] + 1
#             }

#     try:
#         result = await selected_tool.ainvoke({})

#     except Exception as e:
#         result = f"ERROR: {type(e).__name__}: {e}"

#     print("\nTest result:")
#     print(result)

#     test_passed = (
#             "TEST STATUS: PASSED" in str(result)
#             or "5 passed" in str(result)
#         )

#     retry_count = state["retry_count"]

#     if not test_passed:
#         retry_count += 1

#     return {
#             "messages": [
#                 ToolMessage(
#                     content=str(result),
#                     tool_call_id="test_node"
#                 )
#             ],
#             "test_passed": test_passed,
#             "retry_count": retry_count
#         }

async def test_node(state: AgentState):

    print("\n" + "=" * 60, flush=True)
    print("🧪 TEST NODE ENTERED", flush=True)
    print("=" * 60, flush=True)

    selected_tool = tool_map.get("run_tests")

    print(
        f"🔎 run_tests tool = {selected_tool}",
        flush=True
    )

    if selected_tool is None:

        print("❌ run_tests tool not found", flush=True)

        return {
            "messages": [
                ToolMessage(
                    content="ERROR: run_tests tool not found",
                    tool_call_id="test_node"
                )
            ],
            "test_passed": False,
            "retry_count": state["retry_count"] + 1
        }

    print("🚀 Calling run_tests...", flush=True)

    try:

        result = await asyncio.wait_for(
            selected_tool.ainvoke({}),
            timeout=30
        )

    except asyncio.TimeoutError:

        print(
            "⏰ run_tests TIMED OUT after 30 seconds",
            flush=True
        )

        result = (
            "TEST STATUS: TIMEOUT\n"
            "run_tests exceeded 30 seconds."
        )

    except Exception as e:

        print(
            f"❌ run_tests exception: "
            f"{type(e).__name__}: {e}",
            flush=True
        )

        result = (
            f"ERROR: {type(e).__name__}: {e}"
        )

    print("\nTest result:", flush=True)
    print(result, flush=True)

    test_passed = (
        "TEST STATUS: PASSED" in str(result)
        or "5 passed" in str(result)
    )

    retry_count = state["retry_count"]

    if not test_passed:
        retry_count += 1

    return {
        "messages": [
            ToolMessage(
                content=str(result),
                tool_call_id="test_node"
            )
        ],
        "test_passed": test_passed,
        "retry_count": retry_count
    }


# async def review_node(state: AgentState):

#     print("\n" + "=" * 60, flush=True)
#     print("🔍 REVIEW NODE", flush=True)
#     print("=" * 60, flush=True)

#     print(
#         "📋 Checking Git diff and repository status...",
#         flush=True
#     )

#     git_diff_tool = tool_map.get("git_diff")
#     git_status_tool = tool_map.get("git_status")

#     results = []

#     if git_diff_tool is None:
#         print("❌ git_diff tool not found", flush=True)
#     else:
#         print("🔎 Calling git_diff...", flush=True)

#         diff = await git_diff_tool.ainvoke({})

#         print("\n📄 Git diff:", flush=True)
#         print(diff, flush=True)

#         results.append(
#             ToolMessage(
#                 content=str(diff),
#                 tool_call_id="git_diff_review"
#             )
#         )

#     if git_status_tool is None:
#         print("❌ git_status tool not found", flush=True)
#     else:
#         print("🔎 Calling git_status...", flush=True)

#         status = await git_status_tool.ainvoke({})

#         print("\n📊 Git status:", flush=True)
#         print(status, flush=True)

#         results.append(
#             ToolMessage(
#                 content=str(status),
#                 tool_call_id="git_status_review"
#             )
#         )

#     return {
#         "messages": results
#     }


def after_tools(state: AgentState):

    if state["ran_tests"]:
        print("\n🧪 run_tests was executed → TEST NODE")
        return "test"

    print("\n🔄 More agent reasoning required → AGENT")
    return "agent"





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
        ran_tests = False

        for call in last_message.tool_calls:

            tool_name = call["name"]
            arguments = call["args"]

            if tool_name == "run_tests":
                ran_tests = True

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
            "messages": results,
            "ran_tests": ran_tests
        }
    async def review_node(state: AgentState):

        print("\n" + "=" * 60, flush=True)
        print("🔍 REVIEW NODE", flush=True)
        print("=" * 60, flush=True)

        print(
            "📋 Checking Git diff and repository status...",
            flush=True
        )

        git_diff_tool = tool_map.get("git_diff")
        git_status_tool = tool_map.get("git_status")

        results = []

        if git_diff_tool is None:
            print(
                "❌ git_diff tool not found",
                flush=True
            )
        else:
            print(
                "🔎 Calling git_diff...",
                flush=True
            )

            diff = await git_diff_tool.ainvoke({})

            print(
                "\n📄 Git diff:",
                flush=True
            )
            print(diff, flush=True)

            results.append(
                ToolMessage(
                    content=str(diff),
                    tool_call_id="git_diff_review"
                )
            )

        if git_status_tool is None:
            print(
                "❌ git_status tool not found",
                flush=True
            )
        else:
            print(
                "🔎 Calling git_status...",
                flush=True
            )

            status = await git_status_tool.ainvoke({})

            print(
                "\n📊 Git status:",
                flush=True
            )
            print(status, flush=True)

            results.append(
                ToolMessage(
                    content=str(status),
                    tool_call_id="git_status_review"
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
    graph.add_node(
        "test",
        test_node
    )
    graph.add_node(
        "verify", 
        verify_node
    )
    
    graph.add_node(
        "review", 
        review_node
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
            "verify": "verify",
        }
    )

    graph.add_conditional_edges(
    "tools",
    after_tools,
    {
        "test": "test",
        "agent": "agent",
    }
)

    graph.add_conditional_edges(
        "test",
        test_router,
        {
            "success": "review",
            "fix": "agent",
            "failed": END,
        }
    )
    graph.add_conditional_edges(
        "verify",
        verify_router,
        {
            "test": "test",
            "fix": "agent",
            "success": END,
        }
    )
    graph.add_edge("review", END)

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
    png_data = graph.get_graph().draw_mermaid_png()

    with open("langgraph_tree.png", "wb") as f:
        f.write(png_data)

    print("Graph saved as langgraph_tree.png")

    # --------------------------------------------------------
    # USER TASK
    # --------------------------------------------------------

    task = """
Inspect the calculator project.

There may be an error in the implementation.
Read the relevant source files and tests.
Identify the problem and fix it.

After making the fix:
1. Run the test suite.
2. Check the Git diff.
3. Do not commit anything.

Report what you changed and whether all tests passed.
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

Git:
- git_status
- git_diff
- git_log
- git_add
- git_commit

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
        ],
        "test_passed": False,
        "retry_count": 0,
        "max_retries": 3,
        "ran_tests": False
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