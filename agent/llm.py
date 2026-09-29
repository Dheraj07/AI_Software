from langchain_ollama import ChatOllama


llm = ChatOllama(
    model="qwen2.5-coder:14b",
    temperature=0,
)


response = llm.invoke(
    "Write a Python function that multiplies two numbers."
)

print(response.content)