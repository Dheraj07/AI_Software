import asyncio
import ast
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, HumanMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict
from langchain_core.tools import StructuredTool
from yaml import warnings
from mcp_client import MCPClient
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

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
    test_output: str
    baseline_git_status: str
    scope_report: str
    scope_warnings: list[str]
    baseline_git_diff: str
    commit_requested: bool
    human_approved: bool
    approved_files: list[str]
    commit_message: str
    commit_result: str
    final_git_status: str
    baseline_git_cached_diff: str

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
        return "validate_scope"

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

        git_diff_tool = tool_map.get("git_diff")
        git_diff_cached_tool = tool_map.get("git_diff_cached")
        git_status_tool = tool_map.get("git_status")

        baseline_diff = "BASELINE_DIFF_UNAVAILABLE"
        baseline_cached_diff = "BASELINE_CACHED_DIFF_UNAVAILABLE"
        baseline_status = "BASELINE_STATUS_UNAVAILABLE"

        if git_diff_tool is not None:
            try:
                raw = await git_diff_tool.ainvoke({})
                baseline_diff = unwrap_git_result(raw)
            except Exception as exc:
                baseline_diff = f"BASELINE_DIFF_ERROR: {exc}"

        if git_diff_cached_tool is not None:
            try:
                raw = await git_diff_cached_tool.ainvoke({})
                baseline_cached_diff = unwrap_git_result(raw)
            except Exception as exc:
                baseline_cached_diff = (
                    f"BASELINE_CACHED_DIFF_ERROR: {exc}"
                )

        if git_status_tool is not None:
            try:
                raw = await git_status_tool.ainvoke({})
                baseline_status = unwrap_git_result(raw)
            except Exception as exc:
                baseline_status = f"BASELINE_CAPTURE_ERROR: {exc}"

        print(
            "BASELINE STATUS:",
            repr(baseline_status[:1000]),
            flush=True,
        )
        print(
            "BASELINE DIFF:",
            repr(baseline_diff[:1000]),
            flush=True,
        )
        print(
            "BASELINE CACHED DIFF:",
            repr(baseline_cached_diff[:1000]),
            flush=True,
        )

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
            "current_step": 0,
            "baseline_git_status": baseline_status,
            "baseline_git_diff": baseline_diff,
            "baseline_git_cached_diff": baseline_cached_diff,
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
    
    
    def parse_git_status(raw_status: str) -> dict:
        """Parse git status --short --branch output."""

        import ast

        # MCP may return a dictionary represented as a string.
        try:
            parsed = ast.literal_eval(raw_status)
            if isinstance(parsed, dict):
                raw_status = str(parsed.get("result", raw_status))
        except (ValueError, SyntaxError):
            pass

        branch = "Unknown"
        modified_files = []
        untracked_files = []
        staged_files = []
        unstaged_files = []

        for line in raw_status.splitlines():
            line = line.rstrip()

            if not line:
                continue

            if line.startswith("## "):
                branch = line[3:].strip()
                continue

            if line.startswith("EXIT CODE:"):
                continue

            if line.startswith("TEST STATUS:"):
                continue

            if len(line) < 3:
                continue

            status = line[:2]
            filepath = line[3:].strip()

            if not filepath:
                continue

            if status == "??":
                untracked_files.append(filepath)
                continue

            modified_files.append(filepath)

            if status[0] != " ":
                staged_files.append(filepath)

            if status[1] != " ":
                unstaged_files.append(filepath)

        return {
            "branch": branch,
            "modified_files": modified_files,
            "untracked_files": untracked_files,
            "staged_files": staged_files,
            "unstaged_files": unstaged_files,
            "working_tree_clean": (
                not modified_files and not untracked_files
            ),
        }

    
    async def final_agent_node(state: AgentState):
        print("\n" + "=" * 60, flush=True)
        print("🧠 FINAL REPORT NODE", flush=True)
        print("=" * 60, flush=True)

        # Obtain the latest Git status.
        git_status_tool = tool_map.get("git_status")

        if git_status_tool is not None:
            try:
                git_status = str(
                    await git_status_tool.ainvoke({})
                )
            except Exception as e:
                git_status = f"Unable to retrieve Git status: {e}"
        else:
            git_status = "Git status tool unavailable"

        git_info = parse_git_status(git_status)

        # Read the verified test state.
        test_passed = state.get("test_passed", False)
        test_output = state.get("test_output", "")
        ran_tests = state.get("ran_tests", False)

        if not test_output:
            for message in reversed(state.get("messages", [])):
                content = getattr(message, "content", "")

                if (
                    isinstance(content, str)
                    and "TEST EXECUTION RESULT:" in content
                ):
                    test_output = content
                    break

        if ran_tests and test_passed:
            test_summary = "PASSED"
        elif ran_tests:
            test_summary = "FAILED"
        else:
            test_summary = "NOT VERIFIED"

        # Read scope validation BEFORE constructing the report.
        scope_report = state.get(
            "scope_report",
            "Scope validation was not run.",
        )

        scope_warnings = state.get("scope_warnings", [])

        if scope_warnings:
            warnings_text = "\n".join(
                f"- {warning}" for warning in scope_warnings
            )
        else:
            warnings_text = "- None reported"

        modified_files = "\n".join(
            "- " + f for f in git_info["modified_files"]
        ) or "- None"

        untracked_files = "\n".join(
            "- " + f for f in git_info["untracked_files"]
        ) or "- None"

        staged_files = "\n".join(
            "- " + f for f in git_info["staged_files"]
        ) or "- None"

        unstaged_files = "\n".join(
            "- " + f for f in git_info["unstaged_files"]
        ) or "- None"
        scope_report = state.get(
    "scope_report",
    "Scope validation was not run.",
)

        # scope_warnings = state.get("scope_warnings", [])

        # Determine the final status using tests AND scope validation.
        if not ran_tests or not test_passed:
            task_status = "Needs attention"
        elif (
            "INCOMPLETE" in scope_report.upper()
            or "NOT RUN" in scope_report.upper()
            or "UNAVAILABLE" in scope_report.upper()
            or scope_warnings
        ):
            task_status = "Completed with warnings"
        else:
            task_status = "Completed"

        report = f"""
    ## Final Software-Engineering Report

    1. **Task Status:** {task_status}

    2. **Test Results:** {test_summary}

    3. **Test Output:**
    {test_output or "No test output recorded."}

    4. **Change-Scope Validation:**
    {scope_report}

    5. **Scope Warnings:**
    {warnings_text}

    6. **Git Branch:** {git_info["branch"]}

    7. **Modified Files:**
    {modified_files}

    8. **Untracked Files:**
    {untracked_files}

    9. **Staged Files:**
    {staged_files}

    10. **Unstaged Files:**
    {unstaged_files}

    11. **Working Tree Clean:** {
        "Yes" if git_info["working_tree_clean"] else "No"
    }

    12. **Git Commit:** Not performed by the reporting node.
    """

        print(report, flush=True)

        return {
            "messages": [
                HumanMessage(content=report)
            ]
        }


        
    
    async def validate_scope_node(state: AgentState):
        git_status_tool = tool_map.get("git_status")
        git_diff_tool = tool_map.get("git_diff")
        git_diff_cached_tool = tool_map.get("git_diff_cached")

        warnings = []

        def unwrap_result(raw):
            """Extract the result string from an MCP tool response."""
            if isinstance(raw, dict):
                if "result" in raw:
                    return str(raw["result"])
                return str(raw)

            text = str(raw)

            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, dict) and "result" in parsed:
                    return str(parsed["result"])
            except (ValueError, SyntaxError):
                pass

            return text

        def parse_status_paths(raw_status):
            """Parse git status --short --branch output."""
            paths = {}

            for line in unwrap_result(raw_status).splitlines():
                if line.startswith("##") or len(line) < 3:
                    continue

                status = line[:2]
                path = line[3:].strip()

                if not path:
                    continue

                if status == "??":
                    paths[path] = {
                        "staged": False,
                        "unstaged": False,
                        "untracked": True,
                    }
                    continue

                if all(char in " MADRCU?!" for char in status):
                    paths[path] = {
                        "staged": status[0] not in (" ", "?"),
                        "unstaged": status[1] not in (" ", "?"),
                        "untracked": False,
                    }

            return paths

        def parse_diff_by_file(raw_diff):
            """Map Git diff sections to paths."""
            diff_text = unwrap_result(raw_diff)
            sections = {}
            current_path = None
            current_lines = []

            def save_section():
                if current_path is not None:
                    sections[current_path] = "\n".join(current_lines)

            for line in diff_text.splitlines():
                if line.startswith("diff --git "):
                    save_section()
                    current_lines = [line]
                    current_path = None

                    header = line[len("diff --git "):]
                    if " b/" in header:
                        current_path = header.split(" b/", 1)[1]
                elif current_lines:
                    current_lines.append(line)

            save_section()
            return sections

        # --------------------------------------------------------
        # 1. Validate the baseline captured by planner_node
        # --------------------------------------------------------

        baseline_status = unwrap_result(
            state.get("baseline_git_status", "")
        )
        baseline_diff = unwrap_result(
            state.get("baseline_git_diff", "")
        )
        baseline_cached_diff = unwrap_result(
            state.get("baseline_git_cached_diff", "")
        )

        if (
            not baseline_status
            or "BASELINE_CAPTURE_ERROR:" in baseline_status
            or baseline_status == "BASELINE_STATUS_UNAVAILABLE"
        ):
            warnings.append(
                "Initial Git status is unavailable or unreliable."
            )

        if (
            not baseline_diff
            or "BASELINE_DIFF_ERROR:" in baseline_diff
            or baseline_diff == "BASELINE_DIFF_UNAVAILABLE"
        ):
            warnings.append(
                "Initial unstaged diff is unavailable or unreliable."
            )

        if (
            not baseline_cached_diff
            or "BASELINE_CACHED_DIFF_ERROR:" in baseline_cached_diff
            or baseline_cached_diff == "BASELINE_CACHED_DIFF_UNAVAILABLE"
        ):
            warnings.append(
                "Initial staged diff is unavailable or unreliable."
            )

        if git_status_tool is None:
            warnings.append("git_status tool is unavailable.")

        if git_diff_tool is None:
            warnings.append("git_diff tool is unavailable.")

        if git_diff_cached_tool is None:
            warnings.append("git_diff_cached tool is unavailable.")

        if warnings:
            return {
                "scope_report": (
                    "CHANGE-SCOPE VALIDATION: INCOMPLETE\n"
                    + "\n".join(f"- {warning}" for warning in warnings)
                ),
                "scope_warnings": warnings,
            }

        # --------------------------------------------------------
        # 2. Capture final Git state
        # --------------------------------------------------------

        try:
            current_status = unwrap_result(
                await git_status_tool.ainvoke({})
            )
            current_diff = unwrap_result(
                await git_diff_tool.ainvoke({})
            )
            current_cached_diff = unwrap_result(
                await git_diff_cached_tool.ainvoke({})
            )
        except Exception as exc:
            return {
                "scope_report": (
                    "CHANGE-SCOPE VALIDATION: INCOMPLETE\n"
                    f"Could not retrieve final Git data: {exc}"
                ),
                "scope_warnings": [
                    "Final Git data could not be retrieved."
                ],
            }

        # Detect error responses from Git MCP, which may return errors
        # as strings instead of raising exceptions.
        for label, output in (
            ("final Git status", current_status),
            ("final unstaged diff", current_diff),
            ("final staged diff", current_cached_diff),
        ):
            if "EXIT CODE:" in output and not output.strip().endswith(
                "EXIT CODE: 0"
            ):
                warnings.append(f"Git command failed: {label}.")

        if warnings:
            return {
                "scope_report": (
                    "CHANGE-SCOPE VALIDATION: INCOMPLETE\n"
                    + "\n".join(f"- {warning}" for warning in warnings)
                ),
                "scope_warnings": warnings,
            }

        # --------------------------------------------------------
        # 3. Compare baseline and final snapshots
        # --------------------------------------------------------

        baseline_paths = parse_status_paths(baseline_status)
        current_paths = parse_status_paths(current_status)

        baseline_diffs = parse_diff_by_file(baseline_diff)
        current_diffs = parse_diff_by_file(current_diff)

        baseline_cached_diffs = parse_diff_by_file(
            baseline_cached_diff
        )
        current_cached_diffs = parse_diff_by_file(
            current_cached_diff
        )

        newly_present = sorted(
            set(current_paths) - set(baseline_paths)
        )
        no_longer_present = sorted(
            set(baseline_paths) - set(current_paths)
        )

        changed_diff_paths = sorted(
            path
            for path in set(baseline_diffs) | set(current_diffs)
            if baseline_diffs.get(path) != current_diffs.get(path)
        )

        changed_staged_paths = sorted(
            path
            for path in (
                set(baseline_cached_diffs)
                | set(current_cached_diffs)
            )
            if (
                baseline_cached_diffs.get(path)
                != current_cached_diffs.get(path)
            )
        )

        staged_paths = sorted(
            path
            for path, details in current_paths.items()
            if details["staged"]
        )

        untracked_paths = sorted(
            path
            for path, details in current_paths.items()
            if details["untracked"]
        )

        # --------------------------------------------------------
        # 4. Generate validation report
        # --------------------------------------------------------

        report_lines = [
            "CHANGE-SCOPE VALIDATION",
            "Validation status: COMPARISON COMPLETE — REVIEW REQUIRED",
            "Comparison: initial versus final Git status, "
            "unstaged diff, and staged diff",
            f"Initial status paths: {len(baseline_paths)}",
            f"Final status paths: {len(current_paths)}",
            "",
            "Newly present paths:",
        ]

        report_lines.extend(
            f"- {path}" for path in newly_present
        ) if newly_present else report_lines.append("- None")

        report_lines.append("")
        report_lines.append("Paths no longer present in Git status:")
        report_lines.extend(
            f"- {path}" for path in no_longer_present
        ) if no_longer_present else report_lines.append("- None")

        report_lines.append("")
        report_lines.append("Files with changed unstaged diff content:")
        report_lines.extend(
            f"- {path}" for path in changed_diff_paths
        ) if changed_diff_paths else report_lines.append("- None detected")

        report_lines.append("")
        report_lines.append("Files with changed staged diff content:")
        report_lines.extend(
            f"- {path}" for path in changed_staged_paths
        ) if changed_staged_paths else report_lines.append("- None detected")

        report_lines.append("")
        report_lines.append("Currently staged paths:")
        report_lines.extend(
            f"- {path}" for path in staged_paths
        ) if staged_paths else report_lines.append("- None")

        report_lines.append("")
        report_lines.append("Currently untracked paths:")
        report_lines.extend(
            f"- {path}" for path in untracked_paths
        ) if untracked_paths else report_lines.append("- None")

        report_lines.extend([
            "",
            "Git snapshot comparison completed.",
            "Task relevance and pre-existing versus agent-introduced "
            "changes still require review.",
        ])

        return {
            "scope_report": "\n".join(report_lines),
            "scope_warnings": [],
        }

    def unwrap_git_result(raw):
        if isinstance(raw, dict):
            if "result" in raw:
                return str(raw["result"])
            return str(raw)

        text = str(raw)

        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict) and "result" in parsed:
                return str(parsed["result"])
        except (ValueError, SyntaxError):
            pass

        return text


    def git_command_succeeded(result):
        return (
            unwrap_git_result(result).strip().endswith("EXIT CODE: 0")
        )


    def get_diff_paths(diff_text):
        paths = set()

        for line in unwrap_git_result(diff_text).splitlines():
            if not line.startswith("diff --git "):
                continue

            header = line[len("diff --git "):]

            if " b/" in header:
                paths.add(header.split(" b/", 1)[1])

        return paths
    async def approval_node(state: AgentState):
        status_tool = tool_map.get("git_status")
        diff_tool = tool_map.get("git_diff")

        if status_tool is None or diff_tool is None:
            return {
                "human_approved": False,
                "commit_result": "Blocked: Git review tools unavailable."
            }

        status = unwrap_git_result(
            await status_tool.ainvoke({})
        )

        git_info = parse_git_status(status)

        # Fail closed if unrelated or untracked changes exist.
        if git_info["staged_files"] or git_info["untracked_files"]:
            return {
                "human_approved": False,
                "commit_result": (
                    "Blocked: staged or untracked files require "
                    "manual review before committing."
                )
            }

        files = sorted(git_info["modified_files"])

        diff = unwrap_git_result(
            await diff_tool.ainvoke({})
        )

        diff_paths = get_diff_paths(diff)

        if not files or not diff_paths or set(files) != diff_paths:
            return {
                "human_approved": False,
                "commit_result": (
                    "Blocked: Git status and diff do not agree "
                    "on the files to commit."
                )
            }

        # Pause LangGraph and wait for a human decision.
        decision = interrupt({
            "type": "commit_approval",
            "files": files,
            "diff": diff,
            "message": (
                "Review these changes. Approve the commit only "
                "if the files and diff are correct."
            )
        })

        if not isinstance(decision, dict) or not decision.get("approved"):
            return {
                "human_approved": False,
                "approved_files": [],
                "commit_result": "Commit rejected by user."
            }

        message = str(decision.get("message", "")).strip()

        if not message:
            return {
                "human_approved": False,
                "approved_files": [],
                "commit_result": "Commit blocked: message is required."
            }

        return {
            "human_approved": True,
            "approved_files": files,
            "commit_message": message,
            "commit_result": "Approved; awaiting commit verification."
        }
    async def commit_node(state: AgentState):
        if not state.get("human_approved"):
            return {
                "commit_result": "Blocked: explicit approval missing."
            }

        add_tool = tool_map.get("git_add")
        cached_diff_tool = tool_map.get("git_diff_cached")
        commit_tool = tool_map.get("git_commit")
        status_tool = tool_map.get("git_status")

        if not all([
            add_tool,
            cached_diff_tool,
            commit_tool,
            status_tool
        ]):
            return {
                "commit_result": "Blocked: required Git tools unavailable."
            }

        files = state.get("approved_files", [])

        if not files:
            return {
                "commit_result": "Blocked: no approved files."
            }

        # Stage only the approved paths.
        add_result = await add_tool.ainvoke({"files": files})

        if not git_command_succeeded(add_result):
            return {
                "commit_result": (
                    f"Staging failed: {unwrap_git_result(add_result)}"
                )
            }

        # Verify exactly which files are staged.
        cached_diff = await cached_diff_tool.ainvoke({})
        staged_paths = get_diff_paths(cached_diff)

        if staged_paths != set(files):
            return {
                "commit_result": (
                    "Blocked: staged files do not match the approved "
                    f"file list. Staged: {sorted(staged_paths)}. "
                    f"Approved: {sorted(files)}. Review the Git index."
                )
            }

        # Commit only after staged-diff verification.
        commit_output = await commit_tool.ainvoke({
            "message": state["commit_message"]
        })

        if not git_command_succeeded(commit_output):
            return {
                "commit_result": (
                    f"Commit failed: {unwrap_git_result(commit_output)}"
                )
            }

        final_status = unwrap_git_result(
            await status_tool.ainvoke({})
        )

        return {
            "commit_result": unwrap_git_result(commit_output),
            "final_git_status": final_status
        }
    def route_after_scope(state: AgentState):
        if not state.get("commit_requested", False):
            return "final"

        if not state.get("ran_tests") or not state.get("test_passed"):
            return "final"

        if state.get("scope_warnings"):
            return "final"

        scope_report = state.get("scope_report", "")

        if "INCOMPLETE" in scope_report:
            return "final"

        # Block commits when the repository already had changes
        # before this agent run began.
        baseline = parse_git_status(
            state.get("baseline_git_status", "")
        )

        if not baseline["working_tree_clean"]:
            return "final"

        return "approval"

    
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

    graph.add_node("validate_scope", validate_scope_node)
    graph.add_node("approval", approval_node)
    graph.add_node("commit", commit_node)
    
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
        "validate_scope": "validate_scope",
    }
) 
    # graph.add_edge("validate_scope", "final")  
    graph.add_conditional_edges(
    "validate_scope",
    route_after_scope,
    {
        "final": "final",
        "approval": "approval",
    }
)

    def route_after_approval(state: AgentState):
        if state.get("human_approved"):
            return "commit"
        return "final"


    graph.add_conditional_edges(
        "approval",
        route_after_approval,
        {
            "commit": "commit",
            "final": "final",
        }
    )

    graph.add_edge("commit", "final")

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
    return graph.compile(checkpointer=MemorySaver())



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
        "ran_tests": False,
        "commit_requested": False,
        "human_approved": False,
        "approved_files": [],
        "commit_message": "",
        "commit_result": "Commit not requested",
        "final_git_status": "",
    }

    print("\nStarting LangGraph agent...")
    

    config = {
    "configurable": {
        "thread_id": "ph-agent-commit-session"
    }
}

    result = await graph.ainvoke(
        initial_state,
        config=config
    )

    while result.get("__interrupt__"):
        for pending in result["__interrupt__"]:
            payload = pending.value

            print("\n" + "=" * 60)
            print("HUMAN COMMIT APPROVAL REQUIRED")
            print("=" * 60)

            print("\nFiles:")
            for path in payload.get("files", []):
                print(f"  - {path}")

            print("\nProposed Git diff:")
            print(payload.get("diff", ""))

        answer = input(
            "\nApprove this commit? Type yes or no: "
        ).strip().lower()

        if answer == "yes":
            message = input("Commit message: ").strip()

            resume_value = {
                "approved": True,
                "message": message
            }
        else:
            resume_value = {"approved": False}

        result = await graph.ainvoke(
            Command(resume=resume_value),
            config=config
        )

    final_state = result

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