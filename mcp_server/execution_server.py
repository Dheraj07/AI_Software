from pathlib import Path
import subprocess
import sys

from mcp.server.mcpserver import MCPServer


# ============================================================
# CONFIGURATION
# ============================================================

WORKSPACE = Path(__file__).resolve().parent.parent


# ============================================================
# MCP SERVER
# ============================================================

server = MCPServer(
    name="execution-server",
    description="MCP server for executing Python code and tests.",
)


# ============================================================
# RUN PYTHON
# ============================================================

@server.tool()
async def run_python(code: str) -> str:
    """
    Execute a Python code snippet inside the workspace.
    """

    print(
        "🚀 RUN_PYTHON TOOL ENTERED",
        file=sys.stderr,
        flush=True,
    )

    print(
        f"Code received: {code}",
        file=sys.stderr,
        flush=True,
    )

    try:

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                code,
            ],
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            timeout=30,
        )

        output = ""

        if result.stdout:
            output += (
                "STDOUT:\n"
                + result.stdout
            )

        if result.stderr:
            output += (
                "\nSTDERR:\n"
                + result.stderr
            )

        output += (
            f"\nEXIT CODE: "
            f"{result.returncode}"
        )

        return output

    except subprocess.TimeoutExpired:

        return (
            "ERROR: Python execution "
            "timed out."
        )

    except Exception as e:

        return (
            f"ERROR: "
            f"{type(e).__name__}: {e}"
        )


# ============================================================
# RUN TESTS
# ============================================================

@server.tool()
async def run_tests() -> str:
    """
    Run the calculator project's pytest test suite.
    """

    print(
        "🧪 RUN_TESTS TOOL ENTERED",
        file=sys.stderr,
        flush=True,
    )

    try:

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "calculator/tests",
            ],
            cwd=WORKSPACE / "workspace",
            capture_output=True,
            text=True,
            timeout=60,
        )

        return (
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}\n"
            f"EXIT CODE: {result.returncode}\n"
            f"TEST STATUS: "
            f"{'PASSED' if result.returncode == 0 else 'FAILED'}"
        )

    except subprocess.TimeoutExpired:

        return (
            "ERROR: Test execution "
            "timed out."
        )

    except Exception as e:

        return (
            f"ERROR: "
            f"{type(e).__name__}: {e}"
        )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print(
        "Starting Execution MCP Server...",
        file=sys.stderr,
        flush=True,
    )

    server.run(
        transport="stdio"
    )