"""
Drug-Dx-QC Sequential Agent

Creates a SequentialAgent that orchestrates three sub-agents:
1. Drug Identifier Agent - Extracts clean drug names from medications
2. Drug Classifier Agent - Maps drugs to ATC codes with ICD-10 enrichment  
3. QC Evaluator Agent - Validates medication-diagnosis alignment

Follows Google ADK pattern from Kaggle Day 1b:
1. Create SequentialAgent with sub_agents
2. Create InMemoryRunner with the SequentialAgent
3. Execute with await runner.run_debug(prompt)
"""

from google.adk.agents import SequentialAgent
from google.adk.runners import InMemoryRunner
from .drug_identifier import create_drug_identifier_agent
from .drug_classifier import create_drug_classifier_agent
from .qc_evaluator import create_qc_evaluator_agent
from config import ATC_DATABASE_PATH


def create_drug_dx_qc_sequential_agent(
    model: str = "gemini-2.0-flash",
    atc_db_path: str = None
) -> SequentialAgent:
    """
    Create the Drug-Dx-QC SequentialAgent.
    
    This agent coordinates three sub-agents that work sequentially:
    1. Drug Identifier - extracts clean drug names
    2. Drug Classifier - classifies drugs with ATC codes
    3. QC Evaluator - validates medication-diagnosis alignment
    
    Args:
        model: Gemini model to use for all sub-agents
        atc_db_path: Path to ATC database (defaults to config.ATC_DATABASE_PATH)
        
    Returns:
        SequentialAgent configured with three sub-agents
        
    Usage:
        agent = create_drug_dx_qc_sequential_agent()
        runner = InMemoryRunner(agent=agent)
        response = await runner.run_debug(prompt)
        
    Pattern: All agents created via create_*_agent() functions (pure function pattern)
    """
    
    # Use config default if not specified
    if atc_db_path is None:
        atc_db_path = ATC_DATABASE_PATH
    
    # Create individual agents using pure function pattern
    # All agents are stateless - created via create_*_agent() functions
    drug_identifier_agent = create_drug_identifier_agent(model=model)
    drug_classifier_agent = create_drug_classifier_agent(model=model)
    qc_evaluator_agent = create_qc_evaluator_agent(model=model)
    
    # Create SequentialAgent
    root_agent = SequentialAgent(
        name="drug_dx_qc_sequential_agent",
        sub_agents=[
            drug_identifier_agent,
            drug_classifier_agent,
            qc_evaluator_agent
        ],
    )
    
    return root_agent


async def run_drug_dx_qc_pipeline(
    prompt: str,
    model: str = "gemini-2.0-flash",
    atc_db_path: str = None
) -> str:
    """
    Run the Drug-Dx-QC pipeline using InMemoryRunner (correct pattern).
    
    This follows the Kaggle Day 1b pattern:
    1. Create SequentialAgent
    2. Create InMemoryRunner
    3. Execute with await runner.run_debug()
    
    Args:
        prompt: Task prompt for the sequential agent
        model: Gemini model to use
        atc_db_path: Path to ATC database
        
    Returns:
        Response text from the agent
        
    Example:
        prompt = '''
        Process these medications and diagnoses:
        - Medications file: data/medications_synthetic.csv
        - Conditions file: data/conditions.csv
        
        Tasks:
        1. Extract drug names from medications
        2. Classify drugs with ATC codes
        3. Validate medication-diagnosis alignment
        '''
        
        response = await run_drug_dx_qc_pipeline(prompt)
    """
    
    # Step 1: Create SequentialAgent
    root_agent = create_drug_dx_qc_sequential_agent(
        model=model,
        atc_db_path=atc_db_path
    )
    
    # Step 2: Create InMemoryRunner (THIS IS THE KEY!)
    runner = InMemoryRunner(agent=root_agent)
    
    # Step 3: Execute with runner.run_debug()
    response = await runner.run_debug(prompt)
    
    # Extract text from response
    response_text = response.text if hasattr(response, 'text') else str(response)
    
    return response_text
