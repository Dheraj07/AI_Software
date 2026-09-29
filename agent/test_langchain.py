import asyncio

from langchain_ollama import ChatOllama


async def main():

    print("Creating ChatOllama...")

    llm = ChatOllama(
        model="qwen2.5-coder:14b",
        temperature=0,
        base_url="http://localhost:11434",
    )

    print("Sending request...")

    response = await llm.ainvoke(
        "Reply with exactly: HELLO"
    )

    print("\nResponse:")
    print(response.content)


if __name__ == "__main__":
    asyncio.run(main())