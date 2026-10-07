# 🤖 Autonomous AI Software Engineering Agent

An **AI-powered software engineering agent** built with **Qwen, LangGraph, and Model Context Protocol (MCP)**.

The agent can inspect a software repository, read and modify source code, execute Python programs, run tests, inspect Git changes, stage files, and create commits through dynamically discovered MCP tools.

The project is designed to evolve toward an **autonomous software engineering workflow** capable of analyzing failures, correcting code, rerunning tests, and managing changes through Git.

---

## 🚀 Overview

Traditional LLM coding assistants primarily generate code or provide suggestions.

This project goes a step further by giving the LLM access to an actual development environment through **MCP servers**.

The architecture separates:

- **Reasoning** → Qwen
- **Agent orchestration** → LangGraph
- **Tool communication** → MCP
- **File operations** → Filesystem MCP
- **Code execution & testing** → Execution MCP
- **Version control** → Git MCP

The agent dynamically discovers available MCP tools and decides which tool to invoke based on the task.

---

## 🏗️ Architecture

```text
                         ┌──────────────────┐
                         │      User        │
                         │     Task         │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │      Qwen        │
                         │   Reasoning LLM  │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │    LangGraph     │
                         │ Agent Workflow   │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │    MCP Client    │
                         │ Dynamic Tool Map │
                         └────────┬─────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
              ▼                   ▼                   ▼
      ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
      │ Filesystem MCP│   │ Execution MCP │   │    Git MCP    │
      ├───────────────┤   ├───────────────┤   ├───────────────┤
      │ list_files    │   │ run_python    │   │ git_status    │
      │ read_file     │   │ run_tests     │   │ git_diff      │
      │ write_file    │   │               │   │ git_log       │
      └───────────────┘   └───────────────┘   │ git_add       │
                                               │ git_commit    │
                                               └───────────────┘
```

---

## ✨ Features

### 📁 Filesystem Operations

The Filesystem MCP server provides:

- List files and directories
- Read source files
- Write and modify files

Available tools:

```text
list_files
read_file
write_file
```

---

### 🐍 Code Execution

The Execution MCP server allows the agent to interact with the Python development environment.

Available tools:

```text
run_python
run_tests
```

The agent can:

1. Execute Python code
2. Run the project's pytest suite
3. Inspect test output
4. Use test results as feedback for subsequent actions

---

### 🔀 Git Integration

The Git MCP server provides version-control capabilities directly to the agent.

Available tools:

```text
git_status
git_diff
git_log
git_add
git_commit
```

This allows the agent to:

- Inspect repository state
- Review changes
- Inspect recent commits
- Stage selected files
- Create commits

---

## 🧠 Dynamic MCP Tool Discovery

One of the important parts of the architecture is that the agent does not hard-code every tool into the LangGraph workflow.

The MCP client connects to multiple MCP servers and dynamically discovers their tools.

For example:

```text
Filesystem MCP
    ↓
list_files
read_file
write_file

Execution MCP
    ↓
run_python
run_tests

Git MCP
    ↓
git_status
git_diff
git_log
git_add
git_commit
```

The MCP client builds a dynamic tool map:

```text
tool name
    ↓
MCP server
    ↓
MCP session
    ↓
actual tool
```

This allows additional MCP servers and tools to be added without fundamentally changing the agent architecture.

---

## 🔄 Agent Workflow

The current agent follows a tool-driven workflow:

```text
User Task
   ↓
Qwen analyzes task
   ↓
Select MCP tool
   ↓
MCP Client
   ↓
MCP Server
   ↓
Tool execution
   ↓
Result returned to Qwen
   ↓
Next action
```

For a coding task, the workflow can look like:

```text
Read source code
      ↓
Understand implementation
      ↓
Modify source code
      ↓
Run tests
      ↓
Inspect Git diff
      ↓
Review changes
```

---

## 🔁 Self-Correction Workflow

The project is being extended toward an autonomous self-correction loop.

The target workflow is:

```text
              ┌──────────────┐
              │ Analyze Task │
              └──────┬───────┘
                     ↓
              ┌──────────────┐
              │ Modify Code  │
              └──────┬───────┘
                     ↓
              ┌──────────────┐
              │  Run Tests   │
              └──────┬───────┘
                     ↓
              ┌──────────────┐
              │ Tests Passed?│
              └──────┬───────┘
                  ┌──┴──┐
                 YES    NO
                  │      │
                  ↓      ↓
             Review    Analyze
              Diff     Failure
                │        │
                ↓        ↓
             Commit    Fix Code
                         │
                         └──────→ Run Tests
```

A retry limit can be used to prevent infinite correction loops.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| Qwen | Reasoning / LLM |
| Ollama | Local LLM execution |
| LangGraph | Agent orchestration |
| LangChain | Tool integration |
| MCP | Tool communication protocol |
| Pytest | Automated testing |
| Git | Version control |

---

## 📂 Project Structure

```text
ai-software-team/
│
├── agent/
│   ├── agent_loop.py
│   ├── coding_agent.py
│   ├── langgraph_agent.py
│   ├── llm.py
│   ├── mcp_client.py
│   ├── mcp_tools.py
│   ├── test_execution_tool.py
│   ├── test_git_tools.py
│   └── ...
│
├── mcp_server/
│   ├── filesystem_server.py
│   ├── execution_server.py
│   └── git_server.py
│
├── workspace/
│   └── calculator/
│       ├── __init__.py
│       ├── utils.py
│       └── tests/
│           └── test_utils.py
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd ai-software-team
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Python

```bash
python --version
```

The project has been developed and tested with Python 3.11.

---

## 🧪 Running the Tests

Run the calculator test suite:

```bash
cd workspace
python -m pytest calculator/tests
```

Expected result:

```text
collected 5 items

calculator/tests/test_utils.py .....     [100%]

5 passed
```

---

## 🔌 Testing MCP Servers

The MCP servers can be tested independently before running the complete agent.

### Filesystem MCP

Verify that the filesystem tools are discovered:

```text
list_files
read_file
write_file
```

### Execution MCP

Verify:

```text
run_python
run_tests
```

### Git MCP

Verify:

```text
git_status
git_diff
git_log
git_add
git_commit
```

The complete system should report:

```text
Discovered MCP tools:
  - list_files
  - read_file
  - write_file
  - run_python
  - run_tests
  - git_status
  - git_diff
  - git_log
  - git_add
  - git_commit
```

---

## ▶️ Running the Agent

From the project root:

```bash
python agent/langgraph_agent.py
```

A successful startup should look similar to:

```text
Connecting to MCP...

Connecting to filesystem MCP server...
Connecting to execution MCP server...
Connecting to git MCP server...

MCP connected.

Discovered MCP tools:
  - list_files
  - read_file
  - write_file
  - run_python
  - run_tests
  - git_status
  - git_diff
  - git_log
  - git_add
  - git_commit

LangGraph compiled.

Starting LangGraph agent...
```

---

## 🧩 Example Agent Task

A typical task can be:

```text
Inspect the calculator project, identify the implementation
problem, fix it, run the tests, and review the Git diff.
Do not commit the changes.
```

The agent can then perform:

```text
read_file
    ↓
write_file
    ↓
run_tests
    ↓
git_diff
```

For a complete Git workflow:

```text
Inspect repository
      ↓
Read source
      ↓
Modify source
      ↓
Run tests
      ↓
Review diff
      ↓
git_add
      ↓
git_commit
```

---

## 🔐 Security Considerations

The Execution MCP server can execute Python code and therefore should only be used in a controlled development environment.

Recommended practices:

- Run the agent inside a dedicated development workspace.
- Avoid giving the agent access to sensitive directories.
- Never expose private credentials through MCP tools.
- Review Git changes before pushing to remote repositories.
- Add confirmation mechanisms before destructive operations.
- Restrict shell/system execution capabilities where appropriate.

---

## 🎯 Project Goals

The long-term objective is to build an AI software engineering system capable of performing a complete development cycle:

```text
Understand Requirement
        ↓
Explore Repository
        ↓
Plan Implementation
        ↓
Modify Code
        ↓
Run Tests
        ↓
Analyze Failures
        ↓
Self-Correct
        ↓
Review Git Diff
        ↓
Commit Changes
```

This transforms the system from a simple code-generation assistant into an **agentic software engineering workflow**.

---

## 📌 Current Development Status

```text
Phase 1 — MCP Foundation              ✅
Phase 2 — Git MCP Integration         ✅
Phase 3 — Autonomous Agent            🚧
          ├── Tool selection          ✅
          ├── Code modification       ✅
          ├── Test execution          ✅
          ├── Git inspection          ✅
          └── Self-correction         🚧
```

---

## 🔮 Future Improvements

Planned improvements include:

- Self-correcting test/fix loops
- Better failure analysis
- Explicit planning and execution nodes
- Human approval before Git commits
- Multi-agent software engineering workflows
- Code review agent
- Automated issue-to-PR workflow
- More MCP servers and developer tools
- Persistent agent memory
- Improved task planning and state management

---

## 👨‍💻 Author

**Dheraj Majeti**

Computer Science & Engineering  
AI / ML / Generative AI / Agentic AI

---

## ⭐ Project Concept

> **Give an LLM the ability to reason about software, interact with its development environment through MCP, execute and validate its own changes, and progressively move toward autonomous software engineering.**
