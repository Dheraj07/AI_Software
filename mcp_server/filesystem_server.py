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

# def safe_path(relative_path: str) -> Path:
#     """
#     Resolve a path while ensuring that it remains
#     inside the workspace directory.
#     """

#     workspace = WORKSPACE.resolve()
#     target = (workspace / relative_path).resolve()

#     # Prevent:
#     # ../../something
#     # /absolute/path
#     # symlink escapes
#     if target != workspace and workspace not in target.parents:
#         raise ValueError(
#             f"Access denied: {relative_path} "
#             "is outside the workspace."
#         )

#     return target
def safe_path(relative_path: str) -> Path:
    """
    Resolve a path inside the workspace.

    The MCP workspace is already rooted at /workspace,
    so paths should normally be relative to it.

    If the agent accidentally prefixes the path with
    'workspace/', remove that redundant prefix.
    """

    workspace = WORKSPACE.resolve()

    relative_path = relative_path.strip()

    # Normalize the common mistake:
    # workspace/calculator/utils.py
    # -> calculator/utils.py
    if relative_path == "workspace":
        relative_path = "."

    elif relative_path.startswith("workspace/"):
        relative_path = relative_path[len("workspace/"):]

    target = (workspace / relative_path).resolve()

    # Security boundary
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
    print(
        f"[DEBUG] Reading: {path}",
        flush=True
    )


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


# ============================================================
# SERVER STARTUP
# ============================================================

async def main():
    await server.run_stdio_async()


if __name__ == "__main__":
    asyncio.run(main())