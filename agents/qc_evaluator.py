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
from .file_io_tools import read_csv_file, write_csv_file, write_dataframe_to_csv

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
# CORE TOOL FUNCTIONS (Used by both Python & Agent Orchestration)
# ============================================================================

def check_range_match_tool(
    actual_icd10_code: str,
    expected_icd10_range: str
) -> Dict:
    """
    Check if an ICD-10 code falls within an expected range.
    
    This is the single source of truth for range checking, used by both:
    - Python orchestration (evaluate_medications calls this directly)
    - Agent orchestration (agent uses this as a tool via wrapper)
    
    Examples:
        check_range_match_tool("I10", "I10-I15") → {'match': True, ...}
        check_range_match_tool("I50.9", "I30-I52") → {'match': True, ...}
        check_range_match_tool("E11.9", "I10-I15") → {'match': False, ...}
    
    Args:
        actual_icd10_code: Actual ICD-10 code (e.g., "I10", "I50.9")
        expected_icd10_range: Expected range (e.g., "I10-I15", "J00-J99")
    
    Returns:
        Dictionary with range match result
    """
    if not actual_icd10_code or not expected_icd10_range or '-' not in expected_icd10_range:
        return {
            'match': False,
            'code': actual_icd10_code,
            'range': expected_icd10_range
        }
    
    # Extract base code (remove decimal part)
    code_base = actual_icd10_code.split('.')[0]
    
    # Parse range
    try:
        range_start, range_end = expected_icd10_range.split('-')
        range_start = range_start.strip()
        range_end = range_end.strip()
        
        # Simple alphabetical comparison works for ICD-10 codes
        match = range_start <= code_base <= range_end
        
        return {
            'match': match,
            'code': actual_icd10_code,
            'range': expected_icd10_range
        }
    except:
        return {
            'match': False,
            'code': actual_icd10_code,
            'range': expected_icd10_range
        }


def check_diagnosis_match_tool(
    expected_icd10_codes: List[str],
    expected_icd10_ranges: List[str],
    actual_icd10_codes: List[str]
) -> Dict:
    """
    Check if actual diagnoses match expected ICD-10 codes or ranges.
    
    This is the single source of truth for diagnosis matching, used by both:
    - Python orchestration (evaluate_medications calls this directly)
    - Agent orchestration (agent uses this as a tool via wrapper)
    
    Args:
        expected_icd10_codes: Specific ICD-10 codes for drug
        expected_icd10_ranges: ICD-10 ranges for drug (e.g., ["I10-I15", "E10-E14"])
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
    
    # Check range matches using check_range_match_tool
    range_matches = []
    for actual_code in actual_icd10_codes:
        for expected_range in expected_icd10_ranges:
            result = check_range_match_tool(actual_code, expected_range)
            if result['match']:
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
# AGENT TOOL WRAPPERS (Convert to JSON strings for ADK)
# ============================================================================

def check_diagnosis_match_tool_wrapper(
    drug_name: str,
    drug_class: str,
    expected_icd10_codes: str,
    expected_icd10_ranges: str,
    actual_icd10_codes: str
) -> str:
    """
    Agent tool wrapper: Check if actual diagnoses match expected ICD-10 codes or ranges.
    
    This is the ONLY tool the agent needs for diagnosis matching.
    Internally calls check_diagnosis_match_tool() which uses check_range_match_tool() for range checking.
    
    Converts string inputs to lists and returns JSON string for agent consumption.
    
    Args:
        drug_name: Name of the drug
        drug_class: ATC drug class
        expected_icd10_codes: Expected ICD-10 codes (semicolon-separated, e.g., "I10;E11.9")
        expected_icd10_ranges: Expected ICD-10 ranges (semicolon-separated, e.g., "I10-I15;E10-E14")
        actual_icd10_codes: Actual patient ICD-10 codes (semicolon-separated)
    
    Returns:
        JSON string with match assessment
    """
    expected_codes = [c.strip() for c in expected_icd10_codes.split(';') if c.strip()]
    expected_ranges = [r.strip() for r in expected_icd10_ranges.split(';') if r.strip()]
    actual = [c.strip() for c in actual_icd10_codes.split(';') if c.strip()]
    
    # Call core tool function (which internally uses check_range_match_tool)
    result = check_diagnosis_match_tool(expected_codes, expected_ranges, actual)
    
    # Add clinical reasoning to result
    if result['status'] == 'PASS':
        if result['match_type'] == 'exact':
            result['reason'] = f"Exact ICD-10 match found: {result['matched_codes']}"
        elif result['match_type'] == 'range':
            result['reason'] = f"Range match found: {result['matched_codes']}"
    else:
        result['reason'] = f"No matching diagnosis found. Expected codes: {expected_codes}, Expected ranges: {expected_ranges}, Actual: {actual}"
    
    return json.dumps(result)


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
    # Create tools from wrapper functions (return JSON strings for agent)
    # Note: check_diagnosis_tool internally uses check_range_match_tool for range checking
    check_diagnosis_tool = FunctionTool(check_diagnosis_match_tool_wrapper)
    
    # File I/O tools
    read_csv_tool = FunctionTool(read_csv_file)
    write_csv_tool = FunctionTool(write_csv_file)
    write_df_tool = FunctionTool(write_dataframe_to_csv)
    
    # Create agent with tools and instruction
    agent = Agent(
        model=model,
        name="qc_evaluator",
        tools=[
            check_diagnosis_tool,
            read_csv_tool,
            write_csv_tool,
            write_df_tool
        ],
        instruction="""
You are a clinical QC evaluator specializing in medication-diagnosis alignment.

Your task is to assess whether prescribed medications are clinically appropriate
for a patient's diagnosed conditions by:
1. Reading the drug classifications CSV (output from drug_classifier agent) which contains:
   - drug_name, atc_code, atc_class, indication, icd10_codes (expected diagnoses)
2. Reading the patient conditions CSV which contains actual ICD-10 diagnoses
3. For each medication, checking if the patient's actual diagnoses match the drug's expected ICD-10 codes
4. Using check_diagnosis_match_tool_wrapper() to validate matches
   - This tool handles both exact code matches AND range matches automatically

File I/O Operations:
- Use read_csv_file() to read drug classifications and conditions CSV files
- Use write_csv_file() to write output CSV (provide CSV-formatted string)
- Use write_dataframe_to_csv() to write output from dictionary format

Input files you'll receive:
- Drug classifications CSV: contains drug_name, atc_code, atc_class, indication, icd10_codes
- Patient conditions CSV: contains patient, encounter, code (ICD-10 diagnoses)

Use check_diagnosis_match_tool_wrapper() for all diagnosis validation - it handles both exact matches and range checking.
Provide clear PASS/FAIL assessments with clinical reasoning.
""",
        output_key="qc_flags"
    )
    
    return agent


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def evaluate_medications(
    medications_file: str,
    conditions_file: str,
    drug_classifications_file: str,
    output_file: str = "output/qc_flags.csv"
) -> pd.DataFrame:
    """
    Evaluate medication-diagnosis alignment for all patients.
    
    Args:
        medications_file: Path to medications CSV
        conditions_file: Path to conditions CSV
        drug_classifications_file: Path to drug classifications CSV (output from drug_classifier)
        output_file: Path to output QC flags CSV
        
    Returns:
        DataFrame with QC evaluation results
    """
    logger.info(f"Starting QC evaluation")
    logger.info(f"📥 Medications: {medications_file}")
    logger.info(f"📥 Conditions: {conditions_file}")
    logger.info(f"📥 Classifications: {drug_classifications_file}")
    logger.info(f"📤 Output: {output_file}")
    
    # Load data
    medications_df = pd.read_csv(medications_file)
    conditions_df = pd.read_csv(conditions_file)
    classifications_df = pd.read_csv(drug_classifications_file)
    
    # Convert classifications to dictionary for fast lookup
    drug_lookup = {}
    for _, row in classifications_df.iterrows():
        drug_name = row['drug_name'].lower().strip()
        # Parse icd10_codes - it may be JSON string or comma-separated
        icd10_codes_raw = row.get('icd10_codes', '')
        if isinstance(icd10_codes_raw, str):
            # Try parsing as JSON first
            try:
                import json
                icd10_codes = json.loads(icd10_codes_raw)
            except:
                # Fall back to comma-separated
                icd10_codes = [code.strip() for code in icd10_codes_raw.split(',') if code.strip()]
        else:
            icd10_codes = []
        
        drug_lookup[drug_name] = {
            'code': row.get('atc_code', 'UNKNOWN'),
            'drug_class': row.get('atc_class', 'Unknown'),
            'indication': row.get('indication', ''),
            'icd10_codes': icd10_codes,
            'indication_icd10_ranges': []  # Not in CSV format
        }
    
    logger.info(f"📊 Loaded {len(medications_df)} medication records")
    logger.info(f"📊 Loaded {len(conditions_df)} condition records")
    logger.info(f"📊 Loaded {len(drug_lookup)} drug classifications")
    
    # Results list
    qc_results = []
    
    # Process each medication record
    for idx, med_row in medications_df.iterrows():
        patient_id = med_row['PATIENT']
        encounter_id = med_row['ENCOUNTER']
        drug_description = med_row['DESCRIPTION']
        
        # Extract drug name
        drug_name = extract_drug_name_regex(drug_description)
        drug_key = drug_name.lower().strip()
        
        # Get drug classification data
        if drug_key not in drug_lookup:
            logger.warning(f"⚠️  Drug not in classifications: {drug_name}")
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
        
        drug_data = drug_lookup[drug_key]
        expected_codes = drug_data.get('icd10_codes', [])
        expected_ranges = drug_data.get('indication_icd10_ranges', [])
        
        # Get patient's diagnoses for this encounter
        encounter_conditions = conditions_df[
            (conditions_df['patient'] == patient_id) &
            (conditions_df['encounter'] == encounter_id)
        ]
        
        actual_codes = encounter_conditions['code'].tolist()
        
        # Check for match using core tool function
        match_result = check_diagnosis_match_tool(
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
    drug_classifications_file: str = None,
    output_file: str = None
) -> pd.DataFrame:
    """
    Main entry point for QC evaluation.
    
    Args:
        medications_file: Path to medications CSV (default: from config.py)
        conditions_file: Path to conditions CSV (default: from config.py)
        drug_classifications_file: Path to drug classifications CSV (default: output/drug_classifications.csv)
        output_file: Path to output QC flags CSV (default: from config.py)
        
    Returns:
        DataFrame with QC evaluation results
    """
    from config import OUTPUT_DIR
    
    # Use config defaults if not specified
    medications_file = medications_file or TEST_MEDICATIONS_FILE
    conditions_file = conditions_file or TEST_CONDITIONS_FILE
    drug_classifications_file = drug_classifications_file or str(OUTPUT_DIR / "drug_classifications.csv")
    output_file = output_file or QC_FLAGS_OUTPUT
    
    print("\n" + "="*70)
    print("🔍 QC EVALUATOR AGENT - Medication-Diagnosis Alignment Check")
    print("="*70)
    
    # Evaluate medications
    results = evaluate_medications(
        medications_file=medications_file,
        conditions_file=conditions_file,
        drug_classifications_file=drug_classifications_file,
        output_file=output_file
    )
    
    return results


if __name__ == "__main__":
    # Test with sample data
    evaluate_qc()
