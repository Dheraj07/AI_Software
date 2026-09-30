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
    name="git-server",
    description="MCP server for interacting with the local Git repository.",
)


# ============================================================
# HELPER
# ============================================================

def run_git_command(*args: str) -> str:

    result = subprocess.run(
        ["git", *args],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
    )

    output = ""

    if result.stdout:
        output += result.stdout

    if result.stderr:
        output += f"\nSTDERR:\n{result.stderr}"

    output += f"\nEXIT CODE: {result.returncode}"

    return output


# ============================================================
# GIT STATUS
# ============================================================

@server.tool()
async def git_status() -> str:
    """
    Show the current Git repository status.
    """

    print(
        "🔍 GIT STATUS",
        file=sys.stderr,
        flush=True,
    )

    return run_git_command(
        "status",
        "--short",
        "--branch",
    )


# ============================================================
# GIT DIFF
# ============================================================

@server.tool()
async def git_diff() -> str:
    """
    Show unstaged changes in the Git repository.
    """

    print(
        "🔍 GIT DIFF",
        file=sys.stderr,
        flush=True,
    )

    return run_git_command(
        "diff",
    )


# ============================================================
# GIT LOG
# ============================================================

@server.tool()
async def git_log() -> str:
    """
    Show recent Git commits.
    """

    print(
        "🔍 GIT LOG",
        file=sys.stderr,
        flush=True,
    )

    return run_git_command(
        "log",
        "--oneline",
        "-10",
    )

@server.tool()
async def git_add(files: list[str]) -> str:
    """
    Stage files for the next Git commit.
    """

    print(
        f"📦 GIT ADD: {files}",
        file=sys.stderr,
        flush=True,
    )

    return run_git_command(
        "add",
        "--",
        *files,
    )

@server.tool()
async def git_commit(message: str) -> str:
    """
    Create a Git commit with the provided commit message.
    """

    print(
        f"📝 GIT COMMIT: {message}",
        file=sys.stderr,
        flush=True,
    )

    return run_git_command(
        "commit",
        "-m",
        message,
    )

# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print(
        "Starting Git MCP Server...",
        file=sys.stderr,
        flush=True,
    )

    server.run(
        transport="stdio"
    )