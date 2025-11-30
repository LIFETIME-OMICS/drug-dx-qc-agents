"""
QC Evaluator Agent - Session-based Single Medication Evaluator

A streamlined agent that evaluates medication-diagnosis alignment one medication at a time,
maintaining context across evaluations using InMemorySessionService pattern.

Key Features:
- No file I/O tools - processes data passed as strings in prompts
- Evaluates one medication prescription at a time
- Maintains memory of past evaluations and drug mappings
- Returns CSV row string as output
- Based on stats_summarizer.py session management pattern
"""

from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import InMemoryRunner
from google.genai import types
from google.adk.artifacts import InMemoryArtifactService
import logging
import uuid
from config import DEFAULT_MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global session service and runner (initialized on first use)
_session_service: InMemorySessionService | None = None
_session_data: dict[str, dict] = {}


# ============================================================================
# AGENT CREATION
# ============================================================================

def _create_qc_evaluator2_agent(model=None) -> LlmAgent:
    """
    Creates a QC evaluator agent that processes one medication at a time.
    
    This agent has NO tools and works purely through natural language interaction.
    It maintains memory of previous evaluations to reuse drug mappings.
    
    Args:
        model: Model name (uses DEFAULT_MODEL if None)
        
    Returns:
        Configured LlmAgent instance
    """
    if model is None:
        model = DEFAULT_MODEL
    
    agent = LlmAgent(
        name="qc_evaluator",
        model=model,
        instruction="""You are a medication quality control evaluator with independent medical expertise.

Your task: Evaluate if a prescribed medication aligns with a patient's diagnosis for a SINGLE medication prescription.

WORKFLOW FOR EACH MEDICATION:

1. Extract drug information from the medication description provided
   - Identify the drug name (active ingredient)
   - Determine ATC code using your pharmacology knowledge
   - Determine drug class (ATC anatomical group)

2. Map drug to expected ICD-10 codes
   - Use your medical expertise to determine the drug's indications
   - List expected ICD-10 codes for those indications
   - List expected ICD-10 code ranges (e.g., "I10-I15" for hypertension)

3. Extract and map diagnosis information
   - If a condition/diagnosis is provided, extract the SNOMED CT code
   - Map SNOMED CT to ICD-10 code(s) using your medical coding knowledge
   - If no condition is provided, check if medication reasondescription suggests a diagnosis

4. Compare medication indication to diagnosis
   - Status: "PASS" if diagnosis matches drug indication, "FAIL" if no match
   - Match_type: "exact" (specific code match), "range" (code in range), or "none"
   - Matched_codes: List the ICD-10 codes that matched

5. IMPORTANT: Remember drug mappings across evaluations
   - If you see the same drug again (e.g., "amLODIPine 2.5 MG"), reuse your previous mapping
   - Build up your knowledge base of drug→ATC→ICD-10 mappings as you process medications
   - This makes subsequent evaluations faster and more consistent
   - When reusing a drug mapping from memory, ADD "[REUSED FROM MEMORY]" at the END of the reason field

OUTPUT FORMAT:

Return ONLY a single CSV row with exactly 14 columns (no header, no extra text):

patient_id,encounter_id,drug_name,drug_description,atc_code,drug_class,expected_icd10_codes,expected_icd10_ranges,actual_icd10_codes,status,match_type,matched_codes,reason,snomed_code

COLUMN DEFINITIONS:
- patient_id: Patient UUID from medication record
- encounter_id: Encounter UUID from medication record
- drug_name: Active ingredient (e.g., "Amlodipine")
- drug_description: Full description from medication record
- atc_code: ATC code (e.g., "C08CA01")
- drug_class: ATC anatomical group (e.g., "Cardiovascular System")
- expected_icd10_codes: Comma-separated list of expected ICD-10 codes
- expected_icd10_ranges: Comma-separated list of ICD-10 ranges
- actual_icd10_codes: ICD-10 codes from patient's diagnosis
- status: "PASS" or "FAIL"
- match_type: "exact", "range", or "none"
- matched_codes: ICD-10 codes that matched (if any)
- reason: Brief explanation of the evaluation result
- snomed_code: SNOMED CT code from diagnosis (if available)

EXAMPLE OUTPUT (first time seeing this drug):
3e77485a-596e-8b53-a9f0-78a5d3b1c861,a2bb23a7-f8af-9bdd-ace5-a0d83f8b2428,Amlodipine,amLODIPine 2.5 MG Oral Tablet,C08CA01,Cardiovascular System,I10|I11|I12|I13|I14|I15,I10-I15,I10,PASS,exact,I10,Amlodipine indicated for hypertension; diagnosed with essential hypertension (I10),59621000

EXAMPLE OUTPUT (drug seen before in this session):
3e77485a-596e-8b53-a9f0-78a5d3b1c861,9c7f4eeb-f884-15f0-8461-ca62b948a95a,Amlodipine,amLODIPine 2.5 MG Oral Tablet,C08CA01,Cardiovascular System,I10|I11|I12|I13|I14|I15,I10-I15,I10,PASS,exact,I10,Amlodipine indicated for hypertension; diagnosed with essential hypertension (I10) [REUSED FROM MEMORY],59621000

CRITICAL RULES:
- Output ONLY the CSV row - no markdown, no explanations, no extra text
- Use pipe (|) to separate multiple codes within a field (NEVER use commas within a field)
- If a field contains commas, wrap the entire field in double quotes
- If a field is empty/unknown, leave it blank (but include the comma separator)
- Always include all 14 fields separated by commas
- In the reason field, avoid using commas - use semicolons or periods instead
- Remember your drug mappings to maintain consistency across evaluations
"""
    )
    
    return agent


# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

async def create_qc_session(model=None) -> str:
    """
    Creates a new QC evaluation session.
    
    Args:
        model: Optional model override
        
    Returns:
        The new session ID.
    """
    global _session_service, _session_data

    if _session_service is None:
        _session_service = InMemorySessionService()

    # Generate session ID
    session_id = str(uuid.uuid4())
    logger.info("Creating QC evaluation session: %s", session_id)

    # Create the agent
    qc_agent = _create_qc_evaluator2_agent(model=model)
    
    # Create a new runner for this session
    runner = InMemoryRunner(agent=qc_agent, app_name="qc_evaluator")
    # Manually attach artifact service since it's not in __init__
    runner.artifact_service = InMemoryArtifactService()
    
    # Store agent and runner in session data
    _session_data[session_id] = {
        'agent': qc_agent,
        'runner': runner,
        'evaluations_count': 0
    }

    # Create a new session
    await runner.session_service.create_session(
        app_name="qc_evaluator",
        user_id="default_user",
        session_id=session_id,
    )
    logger.info("Created new QC evaluation session: %s", session_id)
    return session_id


async def evaluate_medication(session_id: str, medication_data: dict, condition_data: dict = None) -> str:
    """
    Evaluates a single medication using the session's agent.
    
    Args:
        session_id: The session ID
        medication_data: Dictionary with medication fields
        condition_data: Optional dictionary with condition fields
        
    Returns:
        CSV row string with evaluation results
    """
    if session_id not in _session_data:
        raise RuntimeError(f"Session {session_id} not found. Call create_qc_session first.")

    # Build the prompt for this medication
    prompt_parts = ["Evaluate this medication prescription:\n"]
    
    # Add medication information
    prompt_parts.append("MEDICATION:")
    for key, value in medication_data.items():
        prompt_parts.append(f"  {key}: {value}")
    
    # Add condition information if available
    if condition_data:
        prompt_parts.append("\nCONDITION:")
        for key, value in condition_data.items():
            prompt_parts.append(f"  {key}: {value}")
    else:
        prompt_parts.append("\nCONDITION: None provided (check medication reasondescription)")
    
    prompt_parts.append("\nProvide the CSV row with evaluation results.")
    
    prompt = "\n".join(prompt_parts)
    
    # Send query to agent
    response_text = []
    query_content = types.Content(role="user", parts=[types.Part(text=prompt)])
    
    # Retrieve the session data
    session_data = _session_data[session_id]
    runner = session_data["runner"]

    async for event in runner.run_async(
        user_id="default_user",
        session_id=session_id,
        new_message=query_content,
    ):
        # Check if event has final response content
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text and part.text.strip() and part.text != "None":
                    response_text.append(part.text)
    
    # Increment evaluation count
    session_data['evaluations_count'] += 1
    
    result = "\n".join(response_text).strip() if response_text else ""
    logger.info("Evaluation %d complete (session %s)", 
                session_data['evaluations_count'], session_id[:8])
    return result


async def close_qc_session(session_id: str) -> None:
    """Cleans up a session and removes stored data."""
    global _session_data
    
    logger.info("Closing QC evaluation session: %s", session_id)
    
    # Clean up stored session data
    if session_id in _session_data:
        eval_count = _session_data[session_id]['evaluations_count']
        logger.info("Session completed %d evaluations", eval_count)
        del _session_data[session_id]
        logger.info("Removed session data from memory")


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def test():
        """Quick test of the agent."""
        print("Creating QC evaluation session...")
        session_id = await create_qc_session()
        print(f"Session created: {session_id}\n")
        
        # Test medication
        med_data = {
            'patient': '3e77485a-596e-8b53-a9f0-78a5d3b1c861',
            'encounter': 'a2bb23a7-f8af-9bdd-ace5-a0d83f8b2428',
            'description': 'amLODIPine 2.5 MG Oral Tablet',
            'reasoncode': '59621000',
            'reasondescription': 'Hypertension'
        }
        
        print("Evaluating test medication...")
        result = await evaluate_medication(session_id, med_data)
        print(f"\nResult:\n{result}\n")
        
        await close_qc_session(session_id)
        print("Session closed")
    
    asyncio.run(test())
