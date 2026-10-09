import asyncio

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, HumanMessage

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

    plan: list[str]
    current_step: int
    step_completed: bool
    ran_tests: bool
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

# def should_continue(state: AgentState):

#     last_message = state["messages"][-1]

#     if last_message.tool_calls:

#         return "tools"
#     # Do not allow the agent to finish before
#     # the test suite has actually passed.
#     # if not state["test_passed"]:
#     #     print("\n⚠️ Agent attempted to finish before tests passed.")
#     #     print("🧪 Forcing TEST NODE...")
#     #     return "test"

#     return "verify"

def should_continue(state: AgentState):

    messages = state["messages"]
    last_message = messages[-1]

    if getattr(last_message, "tool_calls", None):
        return "tools"

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



def advance_router(state: AgentState):

    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)

    if current_step >= len(plan) - 1:
        return "final"

    return "agent"
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

    
    if test_passed:
        result_message = (
            f"TESTS PASSED.\n\n"
            f"Test execution output:\n{result}\n\n"
            "Continue to the review stage. "
            "Do not modify code unless another issue is identified."
        )
    else:
        result_message = (
            f"TESTS FAILED.\n\n"
            f"Test execution output:\n{result}\n\n"
            "Analyze the failure carefully. Inspect the relevant "
            "source code and tests, fix the implementation, read "
            "the modified file again, and rerun the tests."
        )

    return {
        "messages": [
            HumanMessage(content=result_message)
        ],
        "test_passed": test_passed,
        "test_output": str(result),
        "ran_tests": True,
        "retry_count": retry_count,
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

    # llm_with_tools = llm.bind_tools(tools)
    agent_tool_names = {
    "list_files",
    "read_file",
    "write_file",
    "run_python",
}

    agent_tools = [
        t for t in tools
        if t.name in agent_tool_names
    ]

    llm_with_tools = llm.bind_tools(agent_tools)

    tool_map = {
        tool.name: tool
        for tool in tools
    }

    graph = StateGraph(
        AgentState
    )

    async def planner_node(state: AgentState):

        print("\n" + "=" * 60, flush=True)
        print("🧠 PLANNER NODE", flush=True)
        print("=" * 60, flush=True)

        planning_prompt = HumanMessage(
    content="""
Analyze the user's software engineering task and create a
HIGH-LEVEL execution plan.

Create between 5 and 8 steps maximum.

Each step must represent a meaningful engineering phase,
not an individual tool call.

For example:

1. Inspect the repository and understand the relevant files.
2. Analyze the source code and existing tests.
3. Implement the required changes.
4. Run the test suite.
5. Analyze and fix any test failures.
6. Review the Git diff and verify the changes.
7. Check Git status and prepare the final report.

Important:
- Do NOT create a separate step for every file.
- Do NOT create a separate step for every tool call.
- Do NOT create overly detailed substeps.
- Multiple tool calls may be required to complete one plan step.
- The executor will decide which tools to use.
- Do NOT perform any actions.
- Do NOT call tools.

Return ONLY the numbered execution plan.
"""
)

        messages = [
            *state["messages"],
            planning_prompt
        ]

        response = await llm.ainvoke(messages)

        plan_text = response.content

        print("\n📋 Generated Plan:", flush=True)
        print(plan_text, flush=True)

        plan = []

        for line in str(plan_text).splitlines():
            line = line.strip()

            if line:
                plan.append(line)

        return {
            "messages": [response],
            "plan": plan,
            "current_step": 0
        }

    async def dynamic_agent_node(state: AgentState):

        print("\n" + "=" * 60, flush=True)
        print("🤖 AGENT NODE", flush=True)
        print("=" * 60, flush=True)

        plan = state.get("plan", [])
        current_step = state.get("current_step", 0)

        if plan and current_step < len(plan):
            current_plan_step = plan[current_step]
        else:
            current_plan_step = "Continue the task using the available tools."
        print(
            f"\n📍 Current Plan Step: "
            f"{current_step + 1}/{len(plan)}",
            flush=True
        )

        print(
            f"   {current_plan_step}",
            flush=True
        )


        plan_context = HumanMessage(
            content=f"""
You are executing a software engineering task.

FULL PLAN:
{chr(10).join(
    f"{i + 1}. {step}"
    for i, step in enumerate(plan)
)}

CURRENT STEP:
Step {current_step + 1}: {current_plan_step}

Rules:
- Perform ONLY actions relevant to the current step.
- Do not repeat successful tool calls without a clear reason.
- Never create files or directories unless the task requires them.
- Do not use list_files to investigate files already identified.
- If the current step requires testing, call run_tests directly.
- If the current step requires Git review, do not modify source files.
- Use filesystem paths relative to the workspace root.
- After obtaining the necessary results, stop requesting tools
  and provide a concise summary of the step.
"""

        )

        messages = [
            *state["messages"],
            plan_context
        ]

        response = await llm_with_tools.ainvoke(messages)

        if response.tool_calls:

            print("\nQwen requested:", flush=True)

            for call in response.tool_calls:

                print(
                    f"  🔧 {call['name']}",
                    flush=True
                )

                print(
                    f"     {call['args']}",
                    flush=True
                )

        else:

            print(
                "\nQwen produced final answer.",
                flush=True
            )

        return {
        "messages": [response]
    }
    async def advance_step_node(state: AgentState):

        current_step = state.get("current_step", 0)
        plan = state.get("plan", [])

        # Move to next step
        if current_step < len(plan) - 1:
            current_step += 1

            print(
                f"\n➡️ Advancing to plan step "
                f"{current_step + 1}/{len(plan)}",
                flush=True
            )

            return {
                "current_step": current_step,
                "step_completed": False
            }

        print(
            "\n🏁 All plan steps completed.",
            flush=True
        )

        return {
            "current_step": current_step,
            "step_completed": True
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
    
    async def final_agent_node(state: AgentState):

        print("\n" + "=" * 60, flush=True)
        print("🧠 FINAL AGENT NODE", flush=True)
        print("=" * 60, flush=True)

        final_instruction = HumanMessage(
            content="""
You are now in the final reporting stage.

The software engineering task has completed its execution and review stages.

Using the complete conversation history, especially:
- the original user task
- files inspected
- changes made
- test results
- Git diff
- Git status

produce a concise final software-engineering report.

Your report MUST include:

1. Task status
2. What was changed
3. Test results
4. Git branch
5. Modified files
6. Untracked files
7. Whether the working tree is clean
8. Confirm that no Git commit was created

Do not perform any more tool calls.
Do not modify any files.
Do not invent information that is not present in the conversation.

Return only the final human-readable report.
"""
        )

        messages = [
            *state["messages"],
            final_instruction
        ]

        response = await llm.ainvoke(messages)

        print("\n📝 Final report generated.", flush=True)

        return {
            "messages": [response]
        }

    graph.add_node(
            "planner",
            planner_node
    )

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
    graph.add_node(
        "final",
        final_agent_node
    )
    graph.add_node(
        "advance_step", 
        advance_step_node
    )

    graph.add_edge(
        START,
        "planner"
    )
    graph.add_edge(
        "planner",
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
    "advance_step",
    advance_router,
    {
        "agent": "agent",
        "final": "final",
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
            "success": "review",
        }
    )
#     graph.add_conditional_edges(
#     "review",
#     review_router,
#     {
#         "advance_step": "advance_step",
#         "final": "final",
#     }
# )
    graph.add_edge(
    "review",
    "advance_step"
)
    graph.add_edge("final", END)
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
        "plan": [],
        "current_step": 0,
        "step_completed": False,
        "test_output": "",
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