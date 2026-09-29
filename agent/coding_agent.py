import asyncio
import json

from langchain_ollama import ChatOllama

from mcp_client import MCPClient


class CodingAgent:

    def __init__(self):

        self.llm = ChatOllama(
            model="qwen3:8b",
            temperature=0,
            base_url="http://localhost:11434",
        )

        self.mcp = MCPClient()

    async def run(self, task):

        print("\n🚀 Starting Coding Agent")

        await self.mcp.connect()

        tools = await self.mcp.list_tools()

        print("\nAvailable MCP tools:")

        for tool in tools:
            print(
                f"- {tool.name}: "
                f"{tool.description}"
            )

        tool_descriptions = []

        for tool in tools:

            tool_descriptions.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                }
            )

        system_prompt = """
You are an autonomous software engineering agent.

You can interact with a workspace using MCP tools.

Available tools are provided below.

You MUST respond with valid JSON.

If you need to call a tool, respond exactly:

{
    "name": "tool_name",
    "arguments": {
        "argument": "value"
    }
}

If the task is complete, respond exactly:

{
    "done": true,
    "message": "description of completed work"
}

Rules:

1. Read a file before modifying it.
2. Never invent the contents of a file.
3. Use only the available tools.
4. Make the smallest necessary change.
5. Verify modifications when appropriate.
"""

        conversation = f"""
SYSTEM:

{system_prompt}

TOOLS:

{json.dumps(tool_descriptions, indent=2)}

USER TASK:

{task}
"""

        max_iterations = 8

        for iteration in range(max_iterations):

            print(
                f"\n🧠 Iteration "
                f"{iteration + 1}/{max_iterations}"
            )

            response = await asyncio.wait_for(
                self.llm.ainvoke(conversation),
                timeout=120,
            )

            text = response.content.strip()

            print("\nQWEN:")
            print(text)

            # ------------------------------------------------
            # Parse JSON
            # ------------------------------------------------

            try:

                action = json.loads(text)

            except json.JSONDecodeError:

                print(
                    "\n⚠️ Qwen returned invalid JSON."
                )

                break

            # ------------------------------------------------
            # Task complete
            # ------------------------------------------------

            if action.get("done"):

                print("\n✅ Agent completed task.")

                print(
                    action.get(
                        "message",
                        ""
                    )
                )

                break

            # ------------------------------------------------
            # Tool call
            # ------------------------------------------------

            tool_name = action.get("name")

            arguments = action.get(
                "arguments",
                {}
            )

            if not tool_name:

                print(
                    "\n⚠️ No tool name returned."
                )

                break

            print(
                f"\n🔧 MCP TOOL → {tool_name}"
            )

            print(
                f"Arguments → {arguments}"
            )

            # ------------------------------------------------
            # Validate tool
            # ------------------------------------------------

            available_tool_names = [
                tool.name
                for tool in tools
            ]

            if tool_name not in available_tool_names:

                print(
                    f"\n❌ Unknown tool: "
                    f"{tool_name}"
                )

                break

            # ------------------------------------------------
            # Execute MCP tool
            # ------------------------------------------------

            result = await self.mcp.call_tool(
                tool_name,
                arguments
            )

            if result.is_error:

                tool_result = (
                    f"ERROR: {result.content}"
                )

            elif result.structured_content:

                tool_result = str(
                    result.structured_content
                )

            else:

                tool_result = str(
                    result.content
                )

            print("\n📦 MCP RESULT:")
            print(tool_result)

            # ------------------------------------------------
            # Continue conversation
            # ------------------------------------------------

            conversation = f"""
{system_prompt}

TOOLS:

{json.dumps(tool_descriptions, indent=2)}

USER TASK:

{task}

PREVIOUS QWEN ACTION:

{json.dumps(action, indent=2)}

MCP TOOL RESULT:

{tool_result}

Now decide the next action.

Return ONLY valid JSON.
"""

        else:

            print(
                "\n⚠️ Maximum iterations reached."
            )

        await self.mcp.close()


async def main():

    agent = CodingAgent()

    task = """
Add a function called multiply(a, b)
to calculator/utils.py.

The function must return a * b.
"""

    await agent.run(task)


if __name__ == "__main__":
    asyncio.run(main())