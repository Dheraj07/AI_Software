from pathlib import Path
import subprocess

from mcp.server.mcpserver import MCPServer

import sys
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

    print("\n🐍 Executing Python code...")

    try:

        result = subprocess.run(
            ["python", "-c", code],
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

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "workspace/calculator/tests",
        ],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
    )

    return (
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}\n"
        f"EXIT CODE: {result.returncode}\n"
        f"TEST STATUS: "
        f"{'PASSED' if result.returncode == 0 else 'FAILED'}"
    )


# ============================================================
# START SERVER
# ============================================================

# if __name__ == "__main__":
#     server.run(
#         transport="stdio"
#     )
if __name__ == "__main__":

    import sys

    print(
        "Starting Execution MCP Server...",
        file=sys.stderr,
    )

    server.run(
        transport="stdio"
    )