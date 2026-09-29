import asyncio
from pathlib import Path

from mcp.server.mcpserver import MCPServer


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = PROJECT_ROOT / "workspace"


server = MCPServer(
    name="Filesystem Server",
    version="1.0.0",
    description="Safe filesystem tools for the AI coding agent.",
)


# ============================================================
# Security
# ============================================================

def safe_path(relative_path: str) -> Path:
    """
    Resolve a path while ensuring that it remains
    inside the workspace directory.
    """

    workspace = WORKSPACE.resolve()
    target = (workspace / relative_path).resolve()

    # Prevent:
    # ../../something
    # /absolute/path
    # symlink escapes
    if target != workspace and workspace not in target.parents:
        raise ValueError(
            f"Access denied: {relative_path} "
            "is outside the workspace."
        )

    return target


# ============================================================
# MCP TOOLS
# ============================================================

@server.tool(
    name="list_files",
    description="List files and directories inside the workspace."
)
async def list_files(directory: str = ".") -> list[str]:

    path = safe_path(directory)

    if not path.exists():
        raise ValueError(
            f"Directory does not exist: {directory}"
        )

    if not path.is_dir():
        raise ValueError(
            f"Not a directory: {directory}"
        )

    return sorted(
        item.relative_to(WORKSPACE).as_posix()
        for item in path.iterdir()
    )


@server.tool(
    name="read_file",
    description="Read the contents of a text file inside the workspace."
)
async def read_file(file_path: str) -> str:

    path = safe_path(file_path)

    if not path.exists():
        raise ValueError(
            f"File does not exist: {file_path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Not a file: {file_path}"
        )

    return path.read_text(encoding="utf-8")


@server.tool(
    name="write_file",
    description="Write or replace a text file inside the workspace."
)
async def write_file(
    file_path: str,
    content: str
) -> str:

    path = safe_path(file_path)

    # Create parent directories if needed
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    path.write_text(
        content,
        encoding="utf-8"
    )

    return f"Successfully wrote: {file_path}"

# @server.tool()
# async def run_python(code: str) -> str:
#     """
#     Execute a Python code snippet inside the workspace
#     and return stdout/stderr.
#     """

#     import subprocess

#     try:
#         result = subprocess.run(
#             ["python", "-c", code],
#             cwd=WORKSPACE,
#             capture_output=True,
#             text=True,
#             timeout=30,
#         )

#         output = ""

#         if result.stdout:
#             output += (
#                 "STDOUT:\n"
#                 + result.stdout
#             )

#         if result.stderr:
#             output += (
#                 "\nSTDERR:\n"
#                 + result.stderr
#             )

#         output += (
#             f"\nEXIT CODE: {result.returncode}"
#         )

#         return output

#     except subprocess.TimeoutExpired:

#         return (
#             "ERROR: Python execution timed out."
#         )

#     except Exception as e:

#         return (
#             f"ERROR: {type(e).__name__}: {e}"
#         )
    
# @server.tool()
# async def run_tests() -> str:
#     """
#     Run the project's pytest test suite and return the results.
#     """

#     import subprocess

#     try:
#         result = subprocess.run(
#             ["python", "-m", "pytest"],
#             cwd=WORKSPACE,
#             capture_output=True,
#             text=True,
#             timeout=60,
#         )

#         output = ""

#         if result.stdout:
#             output += (
#                 "STDOUT:\n"
#                 + result.stdout
#             )

#         if result.stderr:
#             output += (
#                 "\nSTDERR:\n"
#                 + result.stderr
#             )

#         output += (
#             f"\nEXIT CODE: {result.returncode}"
#         )

#         if result.returncode == 0:
#             output += "\nTEST STATUS: PASSED"
#         else:
#             output += "\nTEST STATUS: FAILED"

#         return output

#     except subprocess.TimeoutExpired:

#         return (
#             "ERROR: Test execution timed out."
#         )

#     except Exception as e:

#         return (
#             f"ERROR: {type(e).__name__}: {e}"
#         )
# ============================================================
# SERVER STARTUP
# ============================================================

async def main():
    await server.run_stdio_async()


if __name__ == "__main__":
    asyncio.run(main())