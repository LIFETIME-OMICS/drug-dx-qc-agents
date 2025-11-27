# 📘 Google ADK Agent Coding Guide

This guide documents the two agent patterns used in this project.

**CRITICAL**: Choose the right pattern for your use case.

---

## 🏛️ Two Core Architectures

### 1. `InMemoryRunner` Pattern (Stateless Batch Processing)

For batch processing tasks (reading a file, processing data, writing a file), we use a **stateless** agent pattern.

- **What it is**: A lightweight executor for single-turn agent tasks.
- **When to use**: Independent, repeatable tasks that don't need conversation history.
- **Examples**: `drug_identifier`, `drug_classifier`, `qc_evaluator`
- **Key benefit**: Simple, efficient, no state management overhead.

### 2. `SessionService` Pattern (Stateful Interactive Analysis)

For interactive data analysis with multi-turn conversations, we use a **stateful** session pattern.

- **What it is**: A conversation-based executor that maintains context across multiple queries.
- **When to use**: Interactive analysis, follow-up questions, exploratory data work.
- **Examples**: `stats_summarizer` (with Code Execution for pandas/matplotlib)
- **Key benefit**: Agent remembers previous queries and can build on prior analysis.

---

## 📝 The 4-Step Pattern

Every agent file should be structured in these four parts:

### 1. Tool Functions (Pure Python)

These are standard Python functions that perform specific, non-AI tasks.

- **Rule**: They must be "pure" — no `Agent` or `LLM` calls inside.
- **Example** (`drug_identifier.py`):
  ```python
  def get_csv_columns(file_path: str) -> List[str]:
      """Get list of column names from a CSV file."""
      df = pd.read_csv(file_path, nrows=0)
      return df.columns.tolist()

  def validate_drug_extraction(original: str, extracted: str) -> Dict:
      """Validate that drug extraction looks reasonable."""
      # ... validation logic ...
      return {"is_valid": True, "issues": []}
  ```

### 2. Agent Factory Function (`create_*_agent`)

This function assembles your agent. It bundles the LLM, instructions, and tools.

- **Rule**: It should return a configured `google.adk.Agent` instance.
- **Example** (`drug_identifier.py`):
  ```python
  from google.adk import Agent
  from google.adk.tools import FunctionTool

  def create_drug_identifier_agent(model: str = "gemini-2.5-flash") -> Agent:
      """Creates the Drug Identifier Agent."""
      
      # Wrap pure functions into tools the agent can use
      columns_tool = FunctionTool(get_csv_columns)
      validation_tool = FunctionTool(validate_drug_extraction)
      
      agent = Agent(
          model=model,
          name="drug_identifier",
          instruction="""You are a pharmaceutical expert...
          
          Your role is to...
          - Use get_csv_columns() to find the right column.
          - Use validate_drug_extraction() to verify results.
          """,
          tools=[columns_tool, validation_tool]
      )
      return agent
  ```

### 3. Async Processing Function (`*_async`)

This function executes the agent for a specific task using `InMemoryRunner`.

- **Rule**: Must be an `async def` function. It creates the agent, creates the runner, and calls `await runner.run_debug(prompt)`.
- **Example** (`drug_identifier.py`):
  ```python
  from google.adk.runners import InMemoryRunner

  async def extract_and_validate_drug_async(description: str) -> Dict:
      """Use agent to extract and validate a single drug name."""
      
      agent = create_drug_identifier_agent()
      runner = InMemoryRunner(agent=agent)
      
      prompt = f"Extract the drug name from: '{description}'"
      
      response = await runner.run_debug(prompt)
      # ... parse response.text ...
      return {'drug_name': '...', 'is_valid': True}
  ```

### 4. Main Orchestration Function

This is the primary entry point. It handles file I/O and calls the `async` processing function for each item.

- **Rule**: It reads the input data, loops through it, and saves the final results.
- **Example** (`drug_identifier.py`):
  ```python
  import asyncio

  async def process_medications_file(input_file: str, output_file: str):
      """Main entry point to run the drug identification process."""
      df = pd.read_csv(input_file)
      unique_items = df['DESCRIPTION'].unique()
      
      results = []
      for item in unique_items:
          # Call the async function for each item
          result = await extract_and_validate_drug_async(item)
          results.append(result)
          
      pd.DataFrame(results).to_csv(output_file, index=False)

  # Optional: Sync wrapper for convenience
  def process_medications_file_sync(input_file: str, output_file: str):
      """Synchronous wrapper."""
      return asyncio.run(process_medications_file(input_file, output_file))
  ```

---

## 🔗 Multi-Agent Orchestration: SequentialAgent Pattern

While individual agents handle specific tasks, a "root" agent is often needed to orchestrate the overall workflow. For this project, we use the `adk.SequentialAgent` for this purpose.

#### When to Use `SequentialAgent`

The `SequentialAgent` is the ideal choice for **fixed, linear pipelines** where a series of agents must be executed in a specific, unchanging order. The output from each agent is passed as the input to the subsequent agent.

Our drug QC pipeline is a perfect example:
1.  **Input**: Raw medication data.
2.  `drug_identifier_agent`: Extracts drug names.
3.  `drug_classifier_agent`: Classifies the extracted names.
4.  `qc_evaluator_agent`: Evaluates the classified drugs against patient diagnoses.
5.  **Output**: Final QC report.

#### When to Use a Different Pattern (e.g., "Root Coordinator")

A more dynamic "Root Coordinator" or general `OrchestratorAgent` should be used for **non-linear workflows** where the agent must reason and decide which tool or sub-agent to use next. This is suitable for complex, goal-oriented tasks where the sequence of steps is not predetermined. Since our workflow is fixed, the `SequentialAgent` is the correct and more efficient choice.

#### Implementation Example

The implementation in `agents/drug_dx_qc_agents.py` serves as the reference for this pattern.

```python
# agents/drug_dx_qc_agents.py

def create_drug_dx_qc_sequential_agent(
    model: str = "gemini-2.5-flash",
    atc_db_path: str = None
) -> adk.SequentialAgent:
    """Creates a factory that builds and returns the full Drug-Dx QC pipeline."""
    
    # This factory creates the sub-agents internally
    drug_identifier_agent = create_drug_identifier_agent(model=model)
    drug_classifier_agent = create_drug_classifier_agent(model=model)
    qc_evaluator_agent = create_qc_evaluator_agent(model=model)
    
    # Then, it assembles them into a SequentialAgent
    root_agent = adk.SequentialAgent(
        name="drug_dx_qc_sequential_agent",
        sub_agents=[
            drug_identifier_agent,
            drug_classifier_agent,
            qc_evaluator_agent
        ],
    )
    return root_agent
```

---

By following this 4-step pattern for individual agents and the SequentialAgent pattern for orchestration, we ensure all our agents are consistent, robust, and easy to maintain.

### The Orchestrator as a "Meta-Agent"

A key concept is that the main orchestration function (like `classify_single_drug_async`) acts as a **sequential orchestrator** or "meta-agent". While not a formal `SequentialAgent` object, it serves the same purpose with greater flexibility.

- **`SequentialAgent` Object**: Best for rigid, linear workflows (A -> B -> C).
- **Python Function Orchestrator**: Ideal for complex workflows with conditional logic, loops, and branching. Our `classify_single_drug_async` is a perfect example, as its process is more of a **decision tree than a straight sequence**. This pattern provides the power of sequential agentic workflows with the full flexibility of Python.

---

## 🧠 Direct LLM Calls vs. Agent Tool-Use

A critical design pattern in this project is knowing when to use an agent with tools versus when to make a direct call to the LLM.

### The "Tool-Using" vs. "Knowledge-Providing" Mindset

An LLM's behavior is heavily influenced by its instructions.

1.  **Agent's Mindset (The Tool-User)**: When we give an agent a list of tools (like `fetch_atc_from_who`), we put it into a "tool-using" mode. Its primary goal becomes figuring out **which tool to call**. If a prompt asks a question that can't be answered by its tools, the agent will correctly state that it cannot fulfill the request. It is constrained by its tools.

2.  **Direct LLM's Mindset (The Knowledge-Provider)**: When we make a direct call to the LLM (e.g., via `genai.Client()`), we access its pure, "knowledge-providing" mode. Its goal is simply to answer the prompt based on its vast training data, without the constraint of a specific set of tools.

### When to Use Each Pattern

- **Use an Agent with Tools when**:
  1. The task requires interacting with an **external, authoritative data source** (e.g., a website, a database, an API).
  2. The workflow involves a **structured sequence of steps** that can be mapped to specific functions.
  3. You need the agent to **reason about which tool to use** from a given set.
  - **Example**: `create_drug_classifier_agent` is given the `fetch_atc_from_who` tool because we *must* get the ATC code from the official WHO website, not from the LLM's memory (which could be outdated).

- **Use a Direct LLM Call when**:
  1. The task requires accessing the LLM's **general world knowledge** or creative capabilities.
  2. The prompt is a simple, open-ended question that does not map to a specific tool.
  3. You want to **avoid confusing the agent** by asking it a question that its tools cannot answer.
  - **Example**: `suggest_synonyms_async` and `enrich_icd10_async` make direct calls because we are asking for general knowledge (synonyms, common indications) that is not available through a specific tool. This prevents the agent from incorrectly trying (and failing) to use a tool for the task.

This hybrid approach is a sophisticated and robust design pattern that leverages the best of both worlds: the structured, reliable output of tool-based agents and the broad, flexible knowledge of direct LLM access.

---

## 💬 SessionService Pattern (Stateful Interactive Analysis)

For interactive, exploratory data analysis, use the **SessionService** pattern (Kaggle Day 3a).

### When to Use SessionService

Use this pattern when you need:
- **Multi-turn conversations**: User asks follow-up questions
- **Stateful context**: Agent remembers previous queries and analysis
- **Interactive exploration**: User drills down into data iteratively
- **Code Execution**: Agent writes and runs pandas/matplotlib code to analyze data

### The SessionService Pattern (4 Steps)

#### 1. Create Agent with Code Execution

```python
def create_stats_summarizer_agent(model: str = "gemini-2.0-flash-exp") -> Agent:
    """Create interactive analysis agent with Code Execution."""
    agent = Agent(
        model=model,
        name="stats_summarizer",
        instruction="""You are a data analyst.
        
Use Code Execution to analyze data with pandas.
Create tables, charts, and insights.""",
        tools=[
            FunctionTool(load_data),
            FunctionTool(get_summary)
        ],
        config={
            "code_execution": True  # KEY: Enable Code Execution
        }
    )
    return agent
```

#### 2. Create Session Management Functions

```python
from google.adk.sessions import SessionService

_session_service: Optional[SessionService] = None
_session_data: Dict[str, Any] = {}  # Store data by session_id

async def create_analysis_session(data_file: str) -> str:
    """Create a new analysis session."""
    global _session_service
    
    if _session_service is None:
        _session_service = SessionService()
    
    agent = create_stats_summarizer_agent()
    session_id = await _session_service.create_session(agent=agent)
    
    # Load data for this session
    _session_data[session_id] = load_data_from_file(data_file)
    
    return session_id

async def query_session(session_id: str, query: str) -> str:
    """Send a query to the session."""
    response = await _session_service.send(
        session_id=session_id,
        user_message=query
    )
    return response.text

async def close_session(session_id: str) -> None:
    """Close the session and clean up."""
    await _session_service.close_session(session_id=session_id)
    del _session_data[session_id]
```

#### 3. Use Tools to Access Session Data

Your tool functions need access to the session's data:

```python
def get_top_drugs(session_id: str, n: int = 10) -> str:
    """Get top N drugs from session data."""
    if session_id not in _session_data:
        return "No data loaded for this session."
    
    df = _session_data[session_id]['medications']
    top_drugs = df['drug_name'].value_counts().head(n)
    return top_drugs.to_string()
```

#### 4. Interactive Multi-Turn Usage

```python
# Create session
session_id = await create_analysis_session("data/medications.csv")

# Turn 1
response = await query_session(
    session_id,
    "Show me the top 10 most prescribed drugs"
)
print(response)

# Turn 2 - Agent remembers context!
response = await query_session(
    session_id,
    "Now create a bar chart of those top 10"
)
print(response)

# Turn 3 - Follow-up question
response = await query_session(
    session_id,
    "What percentage of total prescriptions do they represent?"
)
print(response)

# Clean up
await close_session(session_id)
```

### Key Differences: InMemoryRunner vs SessionService

| Feature | InMemoryRunner | SessionService |
|---------|----------------|----------------|
| **State** | Stateless | Stateful |
| **Turns** | Single-turn | Multi-turn |
| **Use Case** | Batch processing | Interactive analysis |
| **Context** | None | Remembers conversation |
| **Code Execution** | Not typically used | Perfect for pandas/matplotlib |
| **Example** | Drug classifier | Stats summarizer |

### Example: Stats Summarizer Agent

See `agents/stats_summarizer.py` for a complete implementation of the SessionService pattern with Code Execution for interactive data analysis.

**Usage:**
```python
from agents.stats_summarizer import create_stats_session, query_stats_session

# Create session with data
session_id = await create_stats_session(
    medications_file="data/medications.csv",
    diagnoses_file="data/diagnoses.csv"
)

# Interactive queries
await query_stats_session(session_id, "How many patients are in the data?")
await query_stats_session(session_id, "Show me drug usage by class")
await query_stats_session(session_id, "Create a chart of the top 10 drugs")
```

---

By understanding both patterns, you can choose the right tool for the job: InMemoryRunner for efficient batch processing, and SessionService for rich, interactive analysis.
