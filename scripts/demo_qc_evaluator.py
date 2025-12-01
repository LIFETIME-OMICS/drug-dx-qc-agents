"""
QC Evaluator Demo - Session-based Single Medication Processing

Demonstrates the InMemorySessionService pattern for medication QC evaluation.
Processes one medication at a time, maintaining context across evaluations.

This demo:
1. Loads medications and conditions CSV files
2. Creates a stateful QC evaluation session
3. Iterates through medications one at a time
4. Extracts medication data and corresponding condition (if exists)
5. Sends to agent as prompt (no file tools)
6. Agent returns CSV row string
7. Script writes results sequentially to output file
8. Agent maintains memory of drug mappings for consistency

Usage:
    python scripts/demo_qc_evaluator.py
    python scripts/demo_qc_evaluator.py --medications data/medications_synthetic_short_8.csv --conditions data/conditions_synthetic_short_8.csv --limit 10
"""

import asyncio
import sys
import os
from pathlib import Path
import argparse
import pandas as pd
from datetime import datetime
import csv
import re
import logging
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.qc_evaluator import (
    create_qc_session,
    evaluate_medication,
    close_qc_session
)
from dotenv import load_dotenv
from config import TEST_MEDICATIONS_FILE, TEST_CONDITIONS_FILE, TEST_DRUG_NAMES_FILE, OUTPUT_DIR

load_dotenv()

# Suppress verbose logging from Google API and httpx
logging.getLogger('google_genai').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)


def extract_medication_data(row: pd.Series) -> dict:
    """
    Extract relevant medication fields from a DataFrame row.
    
    Args:
        row: A pandas Series representing one medication record
        
    Returns:
        Dictionary with medication fields
    """
    return {
        'patient': row.get('patient', ''),
        'encounter': row.get('encounter', ''),
        'code': row.get('code', ''),
        'description': row.get('description', ''),
        'reasoncode': row.get('reasoncode', ''),
        'reasondescription': row.get('reasondescription', ''),
        'encounter.reasoncode': row.get('encounter.reasoncode', ''),
        'encounter.reasondescription': row.get('encounter.reasondescription', ''),
        'start': row.get('start', ''),
        'stop': row.get('stop', ''),
    }


def find_matching_condition(patient_id: str, encounter_id: str, conditions_df: pd.DataFrame) -> dict | None:
    """
    Find the condition record matching the patient and encounter.
    
    Args:
        patient_id: Patient UUID
        encounter_id: Encounter UUID
        conditions_df: DataFrame with conditions data
        
    Returns:
        Dictionary with condition fields, or None if not found
    """
    if conditions_df is None or len(conditions_df) == 0:
        return None
    
    # Try to find exact match on patient and encounter
    matches = conditions_df[
        (conditions_df['patient'] == patient_id) & 
        (conditions_df['encounter'] == encounter_id)
    ]
    
    if len(matches) > 0:
        # Return the first matching condition
        row = matches.iloc[0]
        return {
            'patient': row.get('patient', ''),
            'encounter': row.get('encounter', ''),
            'code': row.get('code', ''),
            'condition_description': row.get('condition_description', ''),
            'encounter.reasoncode': row.get('encounter.reasoncode', ''),
            'encounter.reasondescription': row.get('encounter.reasondescription', ''),
            'start': row.get('start', ''),
            'stop': row.get('stop', ''),
        }
    
    # If no exact match, try to find any condition for this patient
    # (some medications may not have a matching encounter in conditions)
    patient_conditions = conditions_df[conditions_df['patient'] == patient_id]
    if len(patient_conditions) > 0:
        # Return the most recent condition for this patient
        row = patient_conditions.iloc[-1]
        return {
            'patient': row.get('patient', ''),
            'encounter': row.get('encounter', ''),
            'code': row.get('code', ''),
            'condition_description': row.get('condition_description', ''),
            'encounter.reasoncode': row.get('encounter.reasoncode', ''),
            'encounter.reasondescription': row.get('encounter.reasondescription', ''),
            'start': row.get('start', ''),
            'stop': row.get('stop', ''),
            'note': '(Different encounter - most recent condition for patient)'
        }
    
    return None


async def main():
    """Run QC Evaluator V2 demo."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='QC Evaluator V2 Demo')
    parser.add_argument('--medications', default=TEST_MEDICATIONS_FILE,
                        help='Path to medications CSV')
    parser.add_argument('--conditions', default=TEST_CONDITIONS_FILE,
                        help='Path to conditions CSV')
    parser.add_argument('--output', default=None,
                        help='Output CSV file path')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of medications to process')
    args = parser.parse_args()

    meds_file = args.medications
    cond_file = args.conditions
    limit = args.limit
    
    # Set output path
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = str(OUTPUT_DIR / f"qc_results_v2_{timestamp}.csv")
    
    # Check API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("[ERROR] GOOGLE_API_KEY not found in environment")
        print("\nSet it in .env file or export it:")
        print("  export GOOGLE_API_KEY='your-key-here'  # Linux/Mac")
        print("  $env:GOOGLE_API_KEY='your-key-here'    # Windows PowerShell")
        return
    
    print("="*80)
    print("QC EVALUATOR DEMO - Session-based Processing")
    print("="*80)
    print("\nThis demo processes medications one at a time with memory retention.")
    print("The agent learns drug mappings and maintains context across evaluations.\n")
    
    # Check input files
    if not Path(meds_file).exists():
        print(f"[ERROR] Medications file not found: {meds_file}")
        return
    
    print(f"Input files:")
    print(f"   - Medications: {meds_file}")
    
    # Load medications
    print("\nLoading medications data...")
    meds_df = pd.read_csv(meds_file)
    print(f"[OK] Loaded {len(meds_df)} medication records")
    
    # Load conditions (optional)
    conditions_df = None
    if Path(cond_file).exists():
        print(f"   - Conditions: {cond_file}")
        print("\nLoading conditions data...")
        conditions_df = pd.read_csv(cond_file)
        print(f"[OK] Loaded {len(conditions_df)} condition records")
    else:
        print(f"\n[WARNING] Conditions file not found: {cond_file}")
        print("Will process medications without condition matching")
    
    # Apply limit if specified
    if limit and limit < len(meds_df):
        meds_df = meds_df.head(limit)
        print(f"\n[INFO] Limited to first {limit} medications")
    
    print(f"\nOutput file: {output_file}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    # Create QC evaluation session
    print("\n" + "="*80)
    print("Creating QC evaluation session...")
    print("="*80)
    session_id = await create_qc_session()
    print(f"[OK] Session created: {session_id}\n")
    
    # Prepare output file with header
    csv_headers = ["patient_id", "encounter_id", "drug_name", "drug_description", "atc_code", "drug_class",
                   "expected_icd10_codes", "expected_icd10_ranges", "actual_icd10_codes",
                   "status", "match_type", "matched_codes", "reason", "snomed_code"]
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
    
    print(f"Processing {len(meds_df)} medications...\n")
    
    # Process medications one at a time
    results_written = 0
    errors = 0
    
    try:
        for idx, row in meds_df.iterrows():
            med_num = idx + 1
            print(f"[{med_num}/{len(meds_df)}] Processing: {row['description']}")
            
            # Extract medication data
            med_data = extract_medication_data(row)
            
            # Find matching condition
            condition_data = find_matching_condition(
                med_data['patient'],
                med_data['encounter'],
                conditions_df
            )
            
            if condition_data:
                print(f"         Found condition: {condition_data.get('condition_description', 'N/A')}")
            else:
                print(f"         No matching condition found")
            
            try:
                # Evaluate this medication
                result_csv = await evaluate_medication(session_id, med_data, condition_data)
                
                # Clean up the result (remove markdown code blocks if present)
                result_csv = result_csv.strip()
                if result_csv.startswith('```'):
                    # Remove markdown code blocks
                    lines = result_csv.split('\n')
                    result_csv = '\n'.join([l for l in lines if not l.startswith('```')])
                    result_csv = result_csv.strip()
                
                # Parse the CSV row and write using proper CSV writer
                if result_csv:
                    # Split the CSV line, handling quoted fields properly
                    try:
                        # Use csv.reader to parse the agent's output
                        csv_lines = list(csv.reader([result_csv]))
                        if csv_lines and len(csv_lines[0]) > 0:
                            row_data = csv_lines[0]
                            
                            # Ensure we have exactly 14 fields
                            if len(row_data) < 14:
                                # Pad with empty strings
                                row_data.extend([''] * (14 - len(row_data)))
                            elif len(row_data) > 14:
                                # Log warning and truncate
                                print(f"         ⚠ Agent returned {len(row_data)} fields, expected 14. Truncating.")
                                row_data = row_data[:14]
                            
                            # Write using csv.writer for proper escaping
                            with open(output_file, 'a', encoding='utf-8', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow(row_data)
                            
                            results_written += 1
                            print(f"         ✓ Result written")
                        else:
                            print(f"         ⚠ Could not parse agent output")
                            errors += 1
                    except Exception as parse_error:
                        print(f"         ✗ CSV parse error: {parse_error}")
                        print(f"         Raw output: {result_csv[:100]}...")
                        errors += 1
                else:
                    print(f"         ⚠ Empty result from agent")
                    errors += 1
                    
            except Exception as e:
                print(f"         ✗ Error: {e}")
                errors += 1
            
            # Small delay between evaluations
            await asyncio.sleep(0.5)
        
        print("\n" + "="*80)
        print("PROCESSING COMPLETE")
        print("="*80)
        print(f"\nResults:")
        print(f"   - Successfully processed: {results_written}/{len(meds_df)}")
        print(f"   - Errors: {errors}")
        print(f"   - Output file: {output_file}")
        
        # Show sample of results
        if results_written > 0:
            print("\nSample results (first 3 rows):")
            print("-"*80)
            results_df = pd.read_csv(output_file)
            print(results_df.head(3).to_string(index=False))
            print("-"*80)
            
            # Summary statistics
            print("\nSummary:")
            if 'status' in results_df.columns:
                status_counts = results_df['status'].value_counts()
                print(f"   - PASS: {status_counts.get('PASS', 0)}")
                print(f"   - FAIL: {status_counts.get('FAIL', 0)}")
            
            # Check for memory reuse
            if 'reason' in results_df.columns:
                memory_reused = results_df['reason'].astype(str).str.contains('REUSED FROM MEMORY', case=False, na=False).sum()
                print(f"   - Drug mappings reused from memory: {memory_reused}")
        
    finally:
        # Clean up session
        print(f"\nClosing session: {session_id}")
        await close_qc_session(session_id)
        print("[OK] Session closed")
    
    print("\n" + "="*80)
    print("Key Features Demonstrated:")
    print("="*80)
    print("1. Session-based processing with memory retention")
    print("2. Agent learns and reuses drug mappings")
    print("3. No file tools - data passed via prompts")
    print("4. One medication at a time for granular control")
    print("5. Sequential writing by script (not agent)")


# ============================================================================
# NON-AGENT QC EVALUATION (Python + LLM API, no agent orchestration)
# ============================================================================

def check_range_match_tool(
    actual_icd10_code: str,
    expected_icd10_range: str
) -> Dict:
    """
    Check if an ICD-10 code falls within an expected range.
    
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
    expected_icd10_codes: list,
    expected_icd10_ranges: list,
    actual_icd10_codes: list
) -> dict:
    """
    Check if actual diagnoses match expected ICD-10 codes or ranges.
    
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


def map_snomed_to_icd10(
    medication_data: dict,
    condition_data: dict,
    model: str = None
) -> list:
    """
    Map SNOMED CT codes to ICD-10 codes using Gemini API.
    
    This function takes SNOMED CT codes paired with their descriptions from 
    medications and conditions files and maps them to ICD-10 diagnosis codes.
    
    Args:
        medication_data: Dict with paired (code, description) tuples:
            - 'reason': (reasoncode, reasondescription)
            - 'encounter.reason': (encounter.reasoncode, encounter.reasondescription)
        condition_data: Dict with paired (code, description) tuples:
            - 'condition': (code, description)
            - 'encounter.reason': (encounter.reasoncode, encounter.reasondescription)
        model: Gemini model to use (defaults to DEFAULT_MODEL from config)
        
    Returns:
        List of ICD-10 codes mapped from SNOMED CT codes
    """
    from config import DEFAULT_MODEL
    if model is None:
        model = DEFAULT_MODEL
    
    # Filter out empty/None/0 values
    def is_valid_value(value) -> bool:
        if value is None or value == '' or value == 0 or value == '0':
            return False
        if pd.isna(value):
            return False
        return True
    
    # Collect all valid code-description pairs
    code_pairs = []
    
    for key, (code, description) in medication_data.items():
        if is_valid_value(code) or is_valid_value(description):
            code_str = str(code) if is_valid_value(code) else "N/A"
            desc_str = str(description) if is_valid_value(description) else "N/A"
            code_pairs.append(f"Medication {key}: Code={code_str}, Description={desc_str}")
    
    for key, (code, description) in condition_data.items():
        if is_valid_value(code) or is_valid_value(description):
            code_str = str(code) if is_valid_value(code) else "N/A"
            desc_str = str(description) if is_valid_value(description) else "N/A"
            code_pairs.append(f"Condition {key}: Code={code_str}, Description={desc_str}")
    
    # If no valid data, return empty list
    if not code_pairs:
        return []
    
    # Build prompt for LLM
    pairs_text = "\n".join(code_pairs)
    prompt = f"""You are a medical coding expert specializing in SNOMED CT to ICD-10 mapping.

You are given clinical codes (likely SNOMED CT) paired with their descriptions. Map these to the appropriate ICD-10 diagnosis codes.

Clinical code-description pairs:
{pairs_text}

Analyze both the codes and descriptions to identify the diagnoses, then return the corresponding ICD-10 codes.

Return ONLY a JSON array of ICD-10 codes (e.g., ["I10", "E11.9", "J44.0"]).
Include all relevant ICD-10 codes that match the described conditions.
If no clear diagnosis can be mapped, return an empty array [].

Response format: ["CODE1", "CODE2", ...]"""
    
    try:
        # Call Gemini API
        import google.genai as genai
        import json
        
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            print("Warning: No GOOGLE_API_KEY found, skipping LLM enrichment")
            return []
        
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=prompt
        )
        
        # Parse JSON response
        response_text = response.text.strip()
        
        # Extract JSON from response (may have markdown code blocks)
        if response_text.startswith("```"):
            # Remove markdown code blocks
            lines = response_text.split('\n')
            response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
            response_text = response_text.strip()
        
        # Parse JSON
        icd10_codes = json.loads(response_text)
        
        if isinstance(icd10_codes, list):
            # Filter to valid ICD-10 format
            valid_codes = [code for code in icd10_codes if isinstance(code, str) and len(code) >= 3]
            return valid_codes
        else:
            print(f"Warning: Unexpected response format: {type(icd10_codes)}")
            return []
            
    except Exception as e:
        print(f"Warning: Error extracting ICD-10 from text: {e}")
        return []


def load_drug_name_mappings(drug_names_file: str) -> dict:
    """
    Load drug name mappings from the drug_identifier agent output.
    
    Args:
        drug_names_file: Path to drug_names_extracted CSV file
        
    Returns:
        Dictionary mapping drug_description -> drug_name
    """
    drug_names_df = pd.read_csv(drug_names_file)
    
    # Create lookup dictionary
    mapping = {}
    for _, row in drug_names_df.iterrows():
        description = row['drug_description'].strip()
        drug_name = row['drug_name'].strip()
        mapping[description] = drug_name
    
    return mapping


def extract_drug_name(drug_description: str, drug_name_mappings: dict) -> str:
    """
    Extract drug name from description using pre-computed mappings from drug_identifier agent.
    
    Args:
        drug_description: Full drug description
        drug_name_mappings: Dictionary from load_drug_name_mappings()
        
    Returns:
        Extracted drug name, or original description if not found
    """
    # Direct lookup
    if drug_description in drug_name_mappings:
        return drug_name_mappings[drug_description]
    
    # Case-insensitive fallback
    description_lower = drug_description.lower()
    for desc, name in drug_name_mappings.items():
        if desc.lower() == description_lower:
            return name
    
    # Not found - return first word as fallback (original behavior)
    return drug_description.split()[0] if drug_description else drug_description


def evaluate_medications(
    medications_file: str,
    conditions_file: str,
    drug_names_file: str,
    atc_database_file: str,
    output_file: str = "output/qc_flags.csv"
) -> pd.DataFrame:
    """
    Evaluate medication-diagnosis alignment for all patients using Python + LLM API.
    No agent orchestration - direct API calls.
    
    Args:
        medications_file: Path to medications CSV
        conditions_file: Path to conditions CSV
        drug_names_file: Path to drug names CSV (output from drug_identifier)
        atc_database_file: Path to ATC database JSON with ICD-10 ranges
        output_file: Path to output QC flags CSV
        
    Returns:
        DataFrame with QC evaluation results
    """
    import json
    
    print(f"Starting QC evaluation")
    print(f"📥 Medications: {medications_file}")
    print(f"📥 Conditions: {conditions_file}")
    print(f"📥 Drug Names: {drug_names_file}")
    print(f"📥 ATC Database: {atc_database_file}")
    print(f"📤 Output: {output_file}")
    
    # Load data
    medications_df = pd.read_csv(medications_file)
    conditions_df = pd.read_csv(conditions_file)
    drug_name_mappings = load_drug_name_mappings(drug_names_file)
    
    # Load ATC database JSON with ICD-10 ranges
    with open(atc_database_file, 'r') as f:
        atc_database = json.load(f)
    
    # Normalize column names to lowercase for consistency
    medications_df.columns = medications_df.columns.str.lower()
    conditions_df.columns = conditions_df.columns.str.lower()
    
    # Convert ATC database to lookup dictionary
    drug_lookup = {}
    for drug_key, drug_info in atc_database.items():
        drug_name = drug_info.get('drug_name', drug_key).lower().strip()
        
        drug_lookup[drug_name] = {
            'code': drug_info.get('code', 'UNKNOWN'),
            'drug_class': drug_info.get('drug_class', 'Unknown'),
            'indication': drug_info.get('indication', ''),
            'icd10_codes': drug_info.get('icd10_codes', []),
            'indication_icd10_ranges': drug_info.get('indication_icd10_ranges', [])
        }
    
    print(f"📊 Loaded {len(medications_df)} medication records")
    print(f"📊 Loaded {len(conditions_df)} condition records")
    print(f"📊 Loaded {len(drug_name_mappings)} drug name mappings")
    print(f"📊 Loaded {len(drug_lookup)} drug classifications")
    
    # Prepare output file - delete existing and write header
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    header_written = False
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"🗑️  Deleted existing output file: {output_file}")
    
    # Statistics counters
    total_processed = 0
    passed_count = 0
    failed_count = 0
    unknown_count = 0
    
    # Process each medication record
    for idx, med_row in medications_df.iterrows():
        patient_id = med_row['patient']
        encounter_id = med_row['encounter']
        drug_description = med_row['description']
        
        # Extract drug name using pre-computed mappings from drug_identifier agent
        drug_name = extract_drug_name(drug_description, drug_name_mappings)
        drug_key = drug_name.lower().strip()
        
        print(f"\n{'='*80}")
        print(f"🔍 Processing Medication #{idx+1}/{len(medications_df)}")
        print(f"   Drug Name: {drug_name}")
        print(f"   Prescription: {drug_description}")
        
        # Get drug classification data
        if drug_key not in drug_lookup:
            print(f"   ⚠️  Status: Drug not in classifications database")
            
            # Write result immediately
            result_row = {
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
                'matched_codes': '',
                'reason': ''
            }
            
            # Append to CSV file
            result_df = pd.DataFrame([result_row])
            if not header_written:
                result_df.to_csv(output_file, mode='w', index=False, header=True)
                header_written = True
            else:
                result_df.to_csv(output_file, mode='a', index=False, header=False)
            
            total_processed += 1
            unknown_count += 1
            continue
        
        drug_data = drug_lookup[drug_key]
        expected_codes = drug_data.get('icd10_codes', [])
        expected_ranges = drug_data.get('indication_icd10_ranges', [])
        
        # Get patient's diagnoses for this encounter
        encounter_conditions = conditions_df[
            (conditions_df['patient'] == patient_id) &
            (conditions_df['encounter'] == encounter_id)
        ]
        
        # Map SNOMED CT codes to ICD-10 using LLM
        # Collect medication SNOMED code-description pairs
        med_data = {
            'reason': (
                med_row.get('reasoncode', ''),
                med_row.get('reasondescription', '')
            ),
            'encounter.reason': (
                med_row.get('encounter.reasoncode', ''),
                med_row.get('encounter.reasondescription', '')
            )
        }
        
        # Collect condition SNOMED code-description pairs
        cond_data = {}
        if not encounter_conditions.empty:
            first_cond = encounter_conditions.iloc[0]
            cond_data = {
                'condition': (
                    first_cond.get('code', ''),
                    first_cond.get('condition_description', '')
                ),
                'encounter.reason': (
                    first_cond.get('encounter.reasoncode', ''),
                    first_cond.get('encounter.reasondescription', '')
                )
            }
        
        # Display condition information
        print(f"   Condition Info:")
        if med_row.get('reasondescription'):
            print(f"      - Medication reason: {med_row.get('reasondescription', '')} (SNOMED: {med_row.get('reasoncode', '')})")
        if med_row.get('encounter.reasondescription'):
            print(f"      - Encounter reason: {med_row.get('encounter.reasondescription', '')} (SNOMED: {med_row.get('encounter.reasoncode', '')})")
        if not encounter_conditions.empty:
            first_cond = encounter_conditions.iloc[0]
            if first_cond.get('condition_description'):
                print(f"      - Condition record: {first_cond.get('condition_description', '')} (SNOMED: {first_cond.get('code', '')})")
        
        # Map SNOMED CT to ICD-10 codes using LLM
        actual_codes = map_snomed_to_icd10(med_data, cond_data)
        
        # Display actual mapped codes
        if actual_codes:
            print(f"   📝 Actual ICD-10 codes: {actual_codes}")
        else:
            print(f"   📝 Actual ICD-10 codes: NONE")
        
        # Display expected codes and ranges from drug classification
        print(f"   Expected ICD-10 codes: {expected_codes}")
        if expected_ranges:
            print(f"   Expected ICD-10 ranges: {expected_ranges}")
        
        # Check for match using core tool function
        match_result = check_diagnosis_match_tool(
            expected_codes,
            expected_ranges,
            actual_codes
        )
        
        # Display result
        if match_result['status'] == 'PASS':
            print(f"   ✅ Result: PASS - Diagnosis matches drug indication")
            if match_result.get('matched_codes'):
                print(f"      Matched codes: {match_result['matched_codes']}")
            passed_count += 1
        else:
            actual_codes_display = actual_codes if actual_codes else []
            print(f"   ❌ Result: FAIL - No matching diagnosis found : actual_icd10_codes:{actual_codes_display}")
            failed_count += 1
        
        # Record result and write immediately to CSV
        result_row = {
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
            'matched_codes': str(match_result['matched_codes']),
            'reason': ''
        }
        
        # Append to CSV file
        result_df = pd.DataFrame([result_row])
        if not header_written:
            result_df.to_csv(output_file, mode='w', index=False, header=True)
            header_written = True
        else:
            result_df.to_csv(output_file, mode='a', index=False, header=False)
        
        total_processed += 1
    
    # Summary statistics
    print(f"\n{'='*70}")
    print(f"✅ QC EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"📊 Total medications evaluated: {total_processed}")
    print(f"✅ PASS (matching diagnosis): {passed_count} ({passed_count/total_processed*100:.1f}%)")
    print(f"❌ FAIL (no matching diagnosis): {failed_count} ({failed_count/total_processed*100:.1f}%)")
    print(f"⚠️  UNKNOWN (drug not in database): {unknown_count}")
    print(f"📤 Results saved to: {output_file}\n")
    
    # Load and return the complete results
    results_df = pd.read_csv(output_file)
    return results_df


def evaluate_qc(
    medications_file: str = None,
    conditions_file: str = None,
    drug_names_file: str = None,
    atc_database_file: str = None,
    output_file: str = None
) -> pd.DataFrame:
    """
    Main entry point for non-agent QC evaluation (Python + LLM API).
    Uses pre-extracted drug names and ATC database with ICD-10 ranges.
    
    Args:
        medications_file: Path to medications CSV (default: from config.py)
        conditions_file: Path to conditions CSV (default: from config.py)
        drug_names_file: Path to drug names CSV from drug_identifier agent (default: from config.py)
        atc_database_file: Path to ATC database JSON with ICD-10 ranges (default: from config.py)
        output_file: Path to output QC flags CSV (default: from config.py)
        
    Returns:
        DataFrame with QC evaluation results
    """
    from config import QC_FLAGS_OUTPUT, TEST_DRUG_NAMES_FILE, ATC_DATABASE_PATH
    from pathlib import Path
    
    # Use config defaults if not specified
    medications_file = medications_file or TEST_MEDICATIONS_FILE
    conditions_file = conditions_file or TEST_CONDITIONS_FILE
    drug_names_file = drug_names_file or TEST_DRUG_NAMES_FILE
    
    # Use examples/atc_database.json if exists, otherwise output/atc_database.json
    if atc_database_file is None:
        examples_atc = Path("examples/atc_database.json")
        if examples_atc.exists():
            atc_database_file = str(examples_atc)
        else:
            atc_database_file = ATC_DATABASE_PATH
    
    output_file = output_file or QC_FLAGS_OUTPUT
    
    print("\n" + "="*70)
    print("🔍 QC EVALUATOR - Medication-Diagnosis Alignment Check (No Agent)")
    print("="*70)
    
    # Evaluate medications
    results = evaluate_medications(
        medications_file=medications_file,
        conditions_file=conditions_file,
        drug_names_file=drug_names_file,
        atc_database_file=atc_database_file,
        output_file=output_file
    )
    
    return results


if __name__ == "__main__":
    asyncio.run(main())
