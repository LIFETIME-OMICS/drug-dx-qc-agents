"""
QC Evaluator Agent (Agent 3 - Pure Function Pattern)

Validates medication-diagnosis alignment using Google ADK and Gemini.

Pure Function Pattern (following drug_identifier.py):
1. Helper functions for ICD-10 range checking
2. create_qc_evaluator_agent() - returns Agent
3. Async processing functions use InMemoryRunner
4. Sync wrappers for backward compatibility

Workflow:
1. Load medications with ATC classifications
2. Load patient diagnoses (conditions)
3. For each medication encounter, check if patient has matching ICD-10 diagnosis
4. Use rule-based matching (optional LLM for complex cases)
5. Output QC report CSV

Input:  medications CSV + conditions CSV + atc_database.json
Output: qc_flags.csv (patient_id, encounter, drug, expected_icd10, actual_icd10, status)
"""

import json
import os
import sys
import pandas as pd
import asyncio
import logging
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
from google.adk import Agent
from google.adk.tools import FunctionTool
from google.adk.models import Gemini
from google.adk.runners import InMemoryRunner
from .drug_extraction_tools import extract_drug_name_regex

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ATC_DATABASE_PATH, QC_FLAGS_OUTPUT, TEST_MEDICATIONS_FILE, TEST_CONDITIONS_FILE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def is_in_icd10_range(icd10_code: str, icd10_range: str) -> bool:
    """
    Check if an ICD-10 code falls within a specified range.
    
    Examples:
        is_in_icd10_range("I10", "I10-I15") → True
        is_in_icd10_range("I50.9", "I30-I52") → True
        is_in_icd10_range("E11.9", "I10-I15") → False
    
    Args:
        icd10_code: ICD-10 code to check (e.g., "I10", "I50.9")
        icd10_range: ICD-10 range (e.g., "I10-I15", "J00-J99")
        
    Returns:
        True if code is in range, False otherwise
    """
    if not icd10_code or not icd10_range or '-' not in icd10_range:
        return False
    
    # Extract base code (remove decimal part)
    code_base = icd10_code.split('.')[0]
    
    # Parse range
    try:
        range_start, range_end = icd10_range.split('-')
        range_start = range_start.strip()
        range_end = range_end.strip()
        
        # Simple alphabetical comparison works for ICD-10 codes
        return range_start <= code_base <= range_end
    except:
        return False


def check_diagnosis_match_tool(
    drug_name: str,
    drug_class: str,
    expected_icd10_codes: str,
    actual_icd10_codes: str
) -> str:
    """
    Tool: Check if actual diagnoses match expected ICD-10 codes for a drug.
    
    Args:
        drug_name: Name of the drug
        drug_class: ATC drug class
        expected_icd10_codes: Expected ICD-10 codes (semicolon-separated)
        actual_icd10_codes: Actual patient ICD-10 codes (semicolon-separated)
    
    Returns:
        JSON string with match assessment
    """
    expected = [c.strip() for c in expected_icd10_codes.split(';') if c.strip()]
    actual = [c.strip() for c in actual_icd10_codes.split(';') if c.strip()]
    
    # Check exact matches
    exact_matches = set(expected) & set(actual)
    
    if exact_matches:
        return json.dumps({
            'status': 'PASS',
            'match_type': 'exact',
            'matched_codes': list(exact_matches),
            'reason': f'Exact ICD-10 match found: {list(exact_matches)}'
        })
    
    return json.dumps({
        'status': 'FAIL',
        'match_type': 'none',
        'matched_codes': [],
        'reason': f'No matching diagnosis found. Expected: {expected}, Actual: {actual}'
    })


def check_range_match_tool(
    actual_icd10_code: str,
    expected_icd10_range: str
) -> str:
    """
    Tool: Check if an ICD-10 code falls within an expected range.
    
    Args:
        actual_icd10_code: Actual ICD-10 code (e.g., "I10")
        expected_icd10_range: Expected range (e.g., "I10-I15")
    
    Returns:
        JSON string with range match result
    """
    match = is_in_icd10_range(actual_icd10_code, expected_icd10_range)
    
    return json.dumps({
        'match': match,
        'code': actual_icd10_code,
        'range': expected_icd10_range
    })


def load_atc_database(atc_db_path: str = None) -> Dict:
    """Load ATC database with ICD-10 mappings."""
    db_path = atc_db_path or ATC_DATABASE_PATH
    if not os.path.exists(db_path):
        logger.warning(f"⚠️  ATC database not found: {db_path}")
        return {}
    
    with open(db_path, 'r') as f:
        return json.load(f)


def check_diagnosis_match(
    expected_icd10_codes: List[str],
    expected_icd10_ranges: List[str],
    actual_icd10_codes: List[str]
) -> Dict:
    """
    Rule-based diagnosis matching (no LLM).
    
    Check if actual diagnoses match expected ICD-10 codes/ranges.
    
    Args:
        expected_icd10_codes: Specific ICD-10 codes for drug
        expected_icd10_ranges: ICD-10 ranges for drug
        actual_icd10_codes: Patient's actual diagnosis codes
        
    Returns:
        Dictionary with match status and details
    """
    # Check exact matches
    exact_matches = set(expected_icd10_codes) & set(actual_icd10_codes)
    
    if exact_matches:
        return {
            'status': 'PASS',
            'match_type': 'exact',
            'matched_codes': list(exact_matches)
        }
    
    # Check range matches
    range_matches = []
    for actual_code in actual_icd10_codes:
        for expected_range in expected_icd10_ranges:
            if is_in_icd10_range(actual_code, expected_range):
                range_matches.append({
                    'code': actual_code,
                    'range': expected_range
                })
    
    if range_matches:
        return {
            'status': 'PASS',
            'match_type': 'range',
            'matched_codes': range_matches
        }
    
    # No match found
    return {
        'status': 'FAIL',
        'match_type': 'none',
        'matched_codes': []
    }


# ============================================================================
# AGENT CREATION
# ============================================================================

def create_qc_evaluator_agent(model: str = "gemini-2.5-flash") -> Agent:
    """
    Create QC Evaluator Agent following pure function pattern.
    
    Agent has tools for diagnosis matching and clinical reasoning.
    
    Args:
        model: Gemini model to use
        
    Returns:
        Configured Google ADK Agent
    """
    # Create tools from functions
    check_diagnosis_tool = FunctionTool(check_diagnosis_match_tool)
    check_range_tool = FunctionTool(check_range_match_tool)
    
    # Create agent with tools and instruction
    agent = Agent(
        model=model,
        name="qc_evaluator",
        tools=[check_diagnosis_tool, check_range_tool],
        instruction="""
You are a clinical QC evaluator specializing in medication-diagnosis alignment.

Your task is to assess whether prescribed medications are clinically appropriate
for a patient's diagnosed conditions by:
1. Checking if ICD-10 codes match expected indications
2. Evaluating clinical appropriateness using medical reasoning
3. Identifying potential prescribing errors or inconsistencies

Use the provided tools to check diagnosis matches and ICD-10 range membership.
Provide clear PASS/FAIL assessments with clinical reasoning.
"""
    )
    
    return agent


# ============================================================================
# AGENT EXECUTION WITH INMEMORYRUNNER
# ============================================================================

async def evaluate_diagnosis_match_async(
    drug_name: str,
    drug_class: str,
    expected_icd10_codes: List[str],
    expected_icd10_ranges: List[str],
    actual_icd10_codes: List[str],
    use_llm: bool = False,
    agent: Agent = None,
    model: str = "gemini-2.5-flash"
) -> Dict:
    """
    LLM-based diagnosis matching using InMemoryRunner pattern.
    
    Use this method when rule-based matching is insufficient and you need
    clinical reasoning to assess medication-diagnosis alignment.
    
    Args:
        drug_name: Name of the drug
        drug_class: ATC drug class
        expected_icd10_codes: Expected ICD-10 codes for this drug
        expected_icd10_ranges: Expected ICD-10 ranges for this drug
        actual_icd10_codes: Patient's actual diagnosis codes
        use_llm: If False, use rule-based matching only
        agent: Optional pre-created agent
        model: Model to use if creating new agent
        
    Returns:
        Dictionary with match status and clinical reasoning
    """
    # Rule-based matching first
    if not use_llm:
        return check_diagnosis_match(
            expected_icd10_codes,
            expected_icd10_ranges,
            actual_icd10_codes
        )
    
    # LLM-based evaluation with clinical reasoning
    if agent is None:
        agent = create_qc_evaluator_agent(model=model)
    
    runner = InMemoryRunner(agent=agent)
    
    prompt = f"""
Evaluate medication-diagnosis alignment:

Drug: {drug_name}
Drug Class: {drug_class}
Expected ICD-10 Codes: {', '.join(expected_icd10_codes) if expected_icd10_codes else 'None'}
Expected ICD-10 Ranges: {', '.join(expected_icd10_ranges) if expected_icd10_ranges else 'None'}
Actual Patient ICD-10 Codes: {', '.join(actual_icd10_codes) if actual_icd10_codes else 'None'}

Use the check_diagnosis_match_tool to assess if this medication is clinically appropriate
for the patient's diagnoses. Consider:
1. Exact ICD-10 code matches
2. Related diagnostic codes that may not match exactly
3. Clinical appropriateness (e.g., is this drug reasonable for these diagnoses?)

Provide your assessment.
"""
    
    response = await runner.run_debug(prompt, quiet=True)
    
    # Parse response for structured data
    response_text = response.generations[0].candidate.content.parts[0].text
    
    # Try to extract structured assessment from response
    if "PASS" in response_text.upper():
        return {
            'status': 'PASS',
            'match_type': 'llm_assessment',
            'matched_codes': actual_icd10_codes,
            'reasoning': response_text
        }
    else:
        return {
            'status': 'FAIL',
            'match_type': 'llm_assessment',
            'matched_codes': [],
            'reasoning': response_text
        }


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def evaluate_medications(
    medications_file: str,
    conditions_file: str,
    atc_database: Dict,
    output_file: str = "output/qc_flags.csv"
) -> pd.DataFrame:
    """
    Evaluate medication-diagnosis alignment for all patients.
    
    Args:
        medications_file: Path to medications CSV
        conditions_file: Path to conditions CSV
        atc_database: Loaded ATC database dictionary
        output_file: Path to output QC flags CSV
        
    Returns:
        DataFrame with QC evaluation results
    """
    logger.info(f"Starting QC evaluation")
    logger.info(f"📥 Medications: {medications_file}")
    logger.info(f"📥 Conditions: {conditions_file}")
    logger.info(f"📤 Output: {output_file}")
    
    # Load data
    medications_df = pd.read_csv(medications_file)
    conditions_df = pd.read_csv(conditions_file)
    
    logger.info(f"📊 Loaded {len(medications_df)} medication records")
    logger.info(f"📊 Loaded {len(conditions_df)} condition records")
    
    # Results list
    qc_results = []
    
    # Process each medication record
    for idx, med_row in medications_df.iterrows():
        patient_id = med_row['PATIENT']
        encounter_id = med_row['ENCOUNTER']
        drug_description = med_row['DESCRIPTION']
        
        # Extract drug name
        drug_name = extract_drug_name_regex(drug_description)
        
        # Get ATC data for drug
        if drug_name not in atc_database:
            logger.warning(f"⚠️  Drug not in ATC database: {drug_name}")
            qc_results.append({
                'patient_id': patient_id,
                'encounter_id': encounter_id,
                'drug_name': drug_name,
                'drug_description': drug_description,
                'atc_code': 'UNKNOWN',
                'drug_class': 'Unknown',
                'expected_icd10_codes': 'UNKNOWN',
                'expected_icd10_ranges': 'UNKNOWN',
                'actual_icd10_codes': '',
                'status': 'UNKNOWN_DRUG',
                'match_type': 'none',
                'matched_codes': ''
            })
            continue
        
        drug_data = atc_database[drug_name]
        expected_codes = drug_data.get('icd10_codes', [])
        expected_ranges = drug_data.get('indication_icd10_ranges', [])
        
        # Get patient's diagnoses for this encounter
        encounter_conditions = conditions_df[
            (conditions_df['patient'] == patient_id) &
            (conditions_df['encounter'] == encounter_id)
        ]
        
        actual_codes = encounter_conditions['code'].tolist()
        
        # Check for match using rule-based matching
        match_result = check_diagnosis_match(
            expected_codes,
            expected_ranges,
            actual_codes
        )
        
        # Record result
        qc_results.append({
            'patient_id': patient_id,
            'encounter_id': encounter_id,
            'drug_name': drug_name,
            'drug_description': drug_description,
            'atc_code': drug_data.get('code', 'UNKNOWN'),
            'drug_class': drug_data.get('drug_class', 'Unknown'),
            'expected_icd10_codes': '; '.join(expected_codes),
            'expected_icd10_ranges': '; '.join(expected_ranges),
            'actual_icd10_codes': '; '.join(actual_codes) if actual_codes else 'NONE',
            'status': match_result['status'],
            'match_type': match_result['match_type'],
            'matched_codes': str(match_result['matched_codes'])
        })
    
    # Create DataFrame
    results_df = pd.DataFrame(qc_results)
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results_df.to_csv(output_file, index=False)
    
    # Summary statistics
    total = len(results_df)
    passed = len(results_df[results_df['status'] == 'PASS'])
    failed = len(results_df[results_df['status'] == 'FAIL'])
    unknown = len(results_df[results_df['status'] == 'UNKNOWN_DRUG'])
    
    logger.info(f"\n{'='*70}")
    logger.info(f"✅ QC EVALUATION COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"📊 Total medications evaluated: {total}")
    logger.info(f"✅ PASS (matching diagnosis): {passed} ({passed/total*100:.1f}%)")
    logger.info(f"❌ FAIL (no matching diagnosis): {failed} ({failed/total*100:.1f}%)")
    logger.info(f"⚠️  UNKNOWN (drug not in database): {unknown}")
    logger.info(f"📤 Results saved to: {output_file}\n")
    
    return results_df


def evaluate_qc(
    medications_file: str = None,
    conditions_file: str = None,
    atc_db_path: str = None,
    output_file: str = None
) -> pd.DataFrame:
    """
    Main entry point for QC evaluation.
    
    Args:
        medications_file: Path to medications CSV (default: from config.py)
        conditions_file: Path to conditions CSV (default: from config.py)
        atc_db_path: Path to ATC database (default: from config.py)
        output_file: Path to output QC flags CSV (default: from config.py)
        
    Returns:
        DataFrame with QC evaluation results
    """
    # Use config defaults if not specified
    medications_file = medications_file or TEST_MEDICATIONS_FILE
    conditions_file = conditions_file or TEST_CONDITIONS_FILE
    atc_db_path = atc_db_path or ATC_DATABASE_PATH
    output_file = output_file or QC_FLAGS_OUTPUT
    
    print("\n" + "="*70)
    print("🔍 QC EVALUATOR AGENT - Medication-Diagnosis Alignment Check")
    print("="*70)
    
    # Load ATC database
    atc_database = load_atc_database(atc_db_path)
    
    # Evaluate medications
    results = evaluate_medications(
        medications_file=medications_file,
        conditions_file=conditions_file,
        atc_database=atc_database,
        output_file=output_file
    )
    
    return results


if __name__ == "__main__":
    # Test with sample data
    evaluate_qc()
