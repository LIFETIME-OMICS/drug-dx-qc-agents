# 📘 Google ADK Agent Coding Guide (InMemoryRunner Pattern)

This guide documents the correct, simplified pattern for creating agents in this project, based on the architecture in `agents/drug_identifier.py`.

**CRITICAL**: All agents in this project MUST follow this pattern.

---

## 🏛️ The Core Architecture: `InMemoryRunner`

For our batch processing tasks (reading a file, processing data, writing a file), we use a **stateless** agent pattern. The `InMemoryRunner` is the perfect tool for this.

- **What it is**: A lightweight executor for single-turn agent tasks.
- **Why we use it**: Our agents don't need to remember past conversations. Each task (like classifying a drug) is independent. This is efficient and simple.
- **What we AVOID**: `Stateful Agents` or `SessionService`. These are for multi-turn chatbots and would add unnecessary complexity here.

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
