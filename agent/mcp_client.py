# import asyncio
# from pathlib import Path

# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client


# class MCPClient:

#     def __init__(self):

#         project_root = Path(__file__).resolve().parent.parent

#         self.server_path = (
#             project_root
#             / "mcp_server"
#             / "filesystem_server.py"
#         )

#         self.session = None
#         self._stdio_context = None
#         self._session_context = None

#     async def connect(self):

#         server_params = StdioServerParameters(
#             command="python",
#             args=[str(self.server_path)],
#         )

#         self._stdio_context = stdio_client(server_params)

#         read, write = await self._stdio_context.__aenter__()

#         self._session_context = ClientSession(
#             read,
#             write
#         )

#         self.session = await self._session_context.__aenter__()

#         await self.session.initialize()

#         print("Connected to MCP server.")

#     async def list_tools(self):

#         response = await self.session.list_tools()

#         return response.tools

#     async def call_tool(self, name, arguments):

#         return await self.session.call_tool(
#             name,
#             arguments
#         )

#     async def close(self):

#         if self._session_context:
#             await self._session_context.__aexit__(
#                 None,
#                 None,
#                 None
#             )

#         if self._stdio_context:
#             await self._stdio_context.__aexit__(
#                 None,
#                 None,
#                 None
#             )

import asyncio

from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.tools import StructuredTool
from pydantic import create_model
from typing import Any
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class MCPClient:

    def __init__(self):
        self.sessions = {}
        self.tool_map = {}
        self.exit_stack = AsyncExitStack()

    async def connect_server(self, name: str, script_path: Path):
        print(f"Connecting to {name} MCP server...", file=sys.stderr)

        server_params = StdioServerParameters(
            command=sys.executable,
            args=[str(script_path)],
        )

        read, write = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        session = await self.exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await session.initialize()
        response = await session.list_tools()

        print(
            f"\n{name} tools:",
            file=sys.stderr,
            flush=True
        )

        for tool in response.tools:
            print(
                f"  - {tool.name}",
                file=sys.stderr,
                flush=True
            )
        self.sessions[name] = session


    async def connect(self):

        await self.connect_server(
            "filesystem",
            PROJECT_ROOT / "mcp_server" / "filesystem_server.py"
        )

        await self.connect_server(
            "execution",
            PROJECT_ROOT / "mcp_server" / "execution_server.py"
        )
        await self.connect_server(
        "git",
        PROJECT_ROOT / "mcp_server" / "git_server.py"
        )
        await self.list_tools()

    async def list_tools(self):

        self.tool_map = {}

        for server_name, session in self.sessions.items():

            response = await session.list_tools()

            for tool in response.tools:

                self.tool_map[tool.name] = {
                    "server": server_name,
                    "tool": tool,
                }

        return self.tool_map

    async def call_tool(
    self,
    tool_name: str,
    arguments: dict,
):

        print(
            f"\n1️⃣ call_tool entered: {tool_name}",
            file=sys.stderr,
            flush=True
        )

        print(
            f"DEBUG tool_map = {self.tool_map}",
            file=sys.stderr,
            flush=True
        )

        print(
            f"DEBUG tool_name = {tool_name}",
            file=sys.stderr,
            flush=True
        )

        if tool_name not in self.tool_map:
    
            raise ValueError(
                f"Unknown MCP tool: {tool_name}"
            )
    
        server_name = self.tool_map[
            tool_name
        ]["server"]
    
        print(
            f"2️⃣ Server selected: {server_name}",
            file=sys.stderr,
            flush=True
        )
        session = self.sessions[
            server_name
        ]

        print(
            f"3️⃣ Session obtained",
            file=sys.stderr,
            flush=True
        )

        print(
            f"4️⃣ Calling session.call_tool()",
            file=sys.stderr,
            flush=True
        )

        result = await session.call_tool(
            tool_name,
            arguments,
        )

        print(
            f"5️⃣ session.call_tool() returned",
            file=sys.stderr,
            flush=True
        )

        return result

    def get_langchain_tools(self):

        tools = []

        def create_tool(
            tool_name,
            mcp_tool,
            args_schema
        ):

            async def execute_tool(**kwargs):

                print(
                    f"\n🔧 MCP {tool_name}"
                )

                print(
                    f"Arguments: {kwargs}"
                )

                result = await self.call_tool(
                    tool_name,
                    kwargs
                )

                if result.is_error:
                    return f"ERROR: {result.content}"

                return str(
                    result.structured_content
                    or result.content
                )

            return StructuredTool.from_function(
                name=mcp_tool.name,
                description=(
                    mcp_tool.description
                    or f"MCP tool: {mcp_tool.name}"
                ),
                coroutine=execute_tool,
                args_schema=args_schema,
            )

        for tool_name, info in self.tool_map.items():

            mcp_tool = info["tool"]

            schema = mcp_tool.input_schema

            properties = schema.get(
                "properties",
                {}
            )

            required = set(
                schema.get(
                    "required",
                    []
                )
            )

            fields = {}

            for field_name, field_info in properties.items():

                field_type = Any

                if field_info.get("type") == "string":
                    field_type = str

                elif field_info.get("type") == "integer":
                    field_type = int

                elif field_info.get("type") == "number":
                    field_type = float

                elif field_info.get("type") == "boolean":
                    field_type = bool

                default = (
                    ...
                    if field_name in required
                    else field_info.get(
                        "default",
                        None
                    )
                )

                fields[field_name] = (
                    field_type,
                    default
                )

            args_schema = create_model(
                f"{mcp_tool.name}Arguments",
                **fields
            )

            tools.append(
                create_tool(
                    tool_name,
                    mcp_tool,
                    args_schema
                )
            )

        return tools

    async def close(self):

        await self.exit_stack.aclose()