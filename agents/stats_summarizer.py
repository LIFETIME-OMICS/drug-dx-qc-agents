from google.adk.agents import LlmAgent
from google.adk.code_executors import BuiltInCodeExecutor
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
import pandas as pd
import logging
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global session service and runner (initialized on first use)
_session_service: InMemorySessionService | None = None
_runner: Runner | None = None
# Store session data file paths
_session_data_files: dict[str, dict[str, str]] = {}


# ============================================================================
# AGENT CREATION
# ============================================================================

def create_stats_summarizer_agent(data_csv_strings: dict[str, str]) -> LlmAgent:
    """Creates a stats summarizer agent that can perform pandas analysis.
    
    Args:
        data_csv_strings: Dictionary with keys 'medications', 'diagnoses', 'qc_flags' 
                         containing the CSV data as strings.
    """
    meds_csv = data_csv_strings['medications']
    diag_csv = data_csv_strings['diagnoses']
    qc_csv = data_csv_strings['qc_flags']
    
    agent = LlmAgent(
        name="stats_summarizer",
        model="gemini-2.5-flash", 
        code_executor=BuiltInCodeExecutor(),
        instruction=f"""You are a data analysis expert. You have been provided with three CSV datasets as strings.

IMPORTANT: At the start of EVERY analysis, you MUST load the data using pd.read_csv with StringIO:

```python
import pandas as pd
from io import StringIO

# Load the datasets from CSV strings
medications_df = pd.read_csv(StringIO('''
{meds_csv}
'''))

diagnoses_df = pd.read_csv(StringIO('''
{diag_csv}
'''))

qc_flags_df = pd.read_csv(StringIO('''
{qc_csv}
'''))
```

Dataset descriptions:
- medications_df: Contains patient-level medication prescriptions (patient ID, drug descriptions, dates, etc.). Multiple rows per patient showing each prescription.
- diagnoses_df: Contains patient-level diagnosis information (patient ID, diagnosis codes, descriptions, dates, etc.). Multiple rows per patient showing each diagnosis.
- qc_flags_df: Contains drug classification information (drug_name, atc_code, drug_class) AND quality control validation results (status, match_type, matched_codes) for each patient.

Your task is to answer questions about these datasets by:
1. Always starting your code by importing pandas, StringIO, and loading the three datasets as shown above
2. Performing pandas analysis operations as requested
3. Creating visualizations when asked using matplotlib or seaborn
4. Providing clear, concise answers based on the actual data

Remember: Load the data at the beginning of EVERY code execution using the CSV strings provided.""",
    )
    return agent


# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

async def create_stats_session(
    medications_file: str, diagnoses_file: str, qc_flags_file: str
) -> str:
    """
    Creates a new analysis session, loading data and initializing the agent runner.

    Args:
        medications_file: Path to the medications CSV file.
        diagnoses_file: Path to the diagnoses CSV file.
        qc_flags_file: Path to the QC flags CSV file.

    Returns:
        The new session ID.
    """
    global _session_service, _runner, _session_data_files

    if _session_service is None:
        _session_service = InMemorySessionService()

    # Generate session ID
    session_id = str(uuid.uuid4())

    # Load and validate data
    try:
        meds_df = pd.read_csv(medications_file)
        diagnoses_df = pd.read_csv(diagnoses_file)
        qc_flags_df = pd.read_csv(qc_flags_file)
        logger.info(
            "Successfully loaded data for new session: %d medications, %d diagnoses, %d qc_flags",
            len(meds_df),
            len(diagnoses_df),
            len(qc_flags_df),
        )
    except Exception as e:
        logger.error("Failed to load data files: %s", e)
        raise

    # Convert dataframes to CSV strings for embedding in agent instruction
    meds_csv = meds_df.to_csv(index=False)
    diag_csv = diagnoses_df.to_csv(index=False)
    qc_csv = qc_flags_df.to_csv(index=False)
    
    # Store data for this session (in case we need it later)
    _session_data_files[session_id] = {
        'medications_df': meds_df,
        'diagnoses_df': diagnoses_df,
        'qc_flags_df': qc_flags_df
    }
    
    logger.info("Converted data to CSV strings for agent instruction")

    # Create the agent with CSV data strings
    summarizer_agent = create_stats_summarizer_agent(
        data_csv_strings={
            'medications': meds_csv,
            'diagnoses': diag_csv,
            'qc_flags': qc_csv,
        }
    )

    # Create a new runner for each session
    _runner = Runner(
        agent=summarizer_agent,
        session_service=_session_service,
        app_name="stats_summarizer",
    )

    # Create a new session
    await _session_service.create_session(
        app_name=_runner.app_name,
        user_id="default_user",
        session_id=session_id,
    )
    logger.info("Created new stats session: %s", session_id)
    return session_id


async def query_stats_session(session_id: str, query: str) -> str:
    """
    Sends a query to a specific session and gets the agent's response.
    
    Returns:
        The agent's text response as a string.
    """
    if not _runner or not _session_service:
        raise RuntimeError("Session not initialized. Call create_stats_session first.")

    response_text = []
    query_content = types.Content(role="user", parts=[types.Part(text=query)])
    
    async for event in _runner.run_async(
        user_id="default_user",
        session_id=session_id,
        new_message=query_content,
    ):
        # Check if event has final response content
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text and part.text.strip() and part.text != "None":
                    response_text.append(part.text)

    return "\n".join(response_text) if response_text else "(No response)"


async def close_stats_session(session_id: str) -> None:
    """Cleans up a session and removes stored data."""
    global _session_data_files
    
    logger.info("Closing stats session: %s", session_id)
    
    # Clean up stored session data
    if session_id in _session_data_files:
        del _session_data_files[session_id]
        logger.info("Removed session data from memory")
