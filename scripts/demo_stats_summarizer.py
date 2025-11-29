"""
Interactive Stats Summarizer Demo

Demonstrates the InMemorySessionService pattern for interactive data analysis.
Based on Google ADK patterns from Kaggle Day 3a/3b notebooks.

This demo:
1. Takes raw input files (medications + diagnoses)
2. Runs the 3-agent pipeline if processed data doesn't exist
3. Creates a stateful session for interactive analysis
4. Runs 5 demo queries showing multi-turn conversation with context

Usage:
    python scripts/demo_stats_summarizer.py
    python scripts/demo_stats_summarizer.py --medications data/medications_synthetic.csv --diagnoses data/conditions_synthetic.csv
"""

import asyncio
import sys
import os
from pathlib import Path
import argparse
import hashlib

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.stats_summarizer import (
    create_stats_session,
    query_stats_session,
    close_stats_session
)
from agents.drug_identifier import process_medications_file
from agents.drug_classifier import process_drug_names_file
from agents.qc_evaluator import evaluate_medications
import pandas as pd
from dotenv import load_dotenv

from tests.conftest import INPUT_DIR, BASELINE_DIR

load_dotenv()


def get_cache_key(meds_file: str, diag_file: str) -> str:
    """Generate a cache key based on input file paths."""
    combined = f"{meds_file}|{diag_file}"
    return hashlib.md5(combined.encode()).hexdigest()[:8]


async def run_pipeline(meds_file: str, diag_file: str, output_dir: str) -> str:
    """
    Run the 3-agent pipeline using file-processing functions.
    
    Each function internally uses agents for reasoning:
    - process_medications_file() uses drug_identifier agent
    - process_drug_names_file() uses drug_classifier agent  
    - evaluate_medications() uses qc_evaluator agent
    
    Returns:
        Path to qc_flags.csv
    """
    print("\n" + "="*80)
    print("RUNNING 3-AGENT PIPELINE")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output file paths
    drug_names_file = os.path.join(output_dir, "drug_names_extracted.csv")
    classifications_file = os.path.join(output_dir, "drug_classifications.csv")
    qc_flags_file = os.path.join(output_dir, "qc_flags.csv")
    
    # Agent 1: Drug Identifier (uses agent internally for reasoning)
    print("\n[1/3] Drug Identifier: Extracting drug names...")
    await process_medications_file(
        input_file=meds_file,
        output_file=drug_names_file,
        model="gemini-2.5-flash"
    )
    print(f"[OK] Saved: {drug_names_file}")
    
    # Agent 2: Drug Classifier (uses agent internally for reasoning)
    print("\n[2/3] Drug Classifier: Classifying drugs to ATC codes...")
    process_drug_names_file(
        input_file=drug_names_file,
        output_file=classifications_file,
        model="gemini-2.5-flash"
    )
    print(f"[OK] Saved: {classifications_file}")
    
    # Agent 3: QC Evaluator (uses agent internally for reasoning)
    print("\n[3/3] QC Evaluator: Validating drug-diagnosis alignment...")
    
    # Build ATC database from classifications for QC evaluator
    classifications_df = pd.read_csv(classifications_file)
    atc_database = {}
    
    for _, row in classifications_df.iterrows():
        drug_name = row['drug_name']
        
        # Parse ICD-10 codes
        icd10_list = []
        if pd.notna(row.get('icd10_codes')) and row['icd10_codes']:
            icd10_list = [c.strip() for c in str(row['icd10_codes']).split(',')]
        
        atc_database[drug_name] = {
            'code': row['atc_code'],
            'drug_class': row.get('atc_class', row.get('drug_class', 'Unknown')),
            'icd10_codes': icd10_list,
            'indication_icd10_ranges': []  # Not available from classifier output
        }
    
    # Run QC evaluation
    evaluate_medications(
        medications_file=meds_file,
        conditions_file=diag_file,
        atc_database=atc_database,
        output_file=qc_flags_file
    )
    print(f"[OK] Saved: {qc_flags_file}")
    print("\n[OK] Pipeline complete!")
    
    return qc_flags_file


async def main():
    """Run interactive demo of Stats Summarizer agent."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Interactive Stats Summarizer Demo')
    parser.add_argument('--medications', default=str(INPUT_DIR / "medications_test.csv"),
                        help='Path to medications CSV')
    parser.add_argument('--diagnoses', default=str(INPUT_DIR / "conditions_test.csv"),
                        help='Path to diagnoses CSV')
    args = parser.parse_args()
    
    meds_file = args.medications
    diag_file = args.diagnoses
    
    # Check API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("[ERROR] GOOGLE_API_KEY not found in environment")
        print("\nSet it in .env file or export it:")
        print("  export GOOGLE_API_KEY='your-key-here'  # Linux/Mac")
        print("  $env:GOOGLE_API_KEY='your-key-here'    # Windows PowerShell")
        return
    
    print("="*80)
    print("INTERACTIVE STATS SUMMARIZER DEMO")
    print("="*80)
    print("\nThis demo shows the SessionService pattern for multi-turn analysis.")
    print("The agent will remember context across queries.\n")
    
    if not Path(meds_file).exists():
        print(f"[ERROR] Input file not found: {meds_file}")
        print("\nPlease provide valid input files.")
        return
    
    if not Path(diag_file).exists():
        print(f"[ERROR] Input file not found: {diag_file}")
        print("\nPlease provide valid input files.")
        return
    
    # Determine output directory based on input files (working directory)
    cache_key = get_cache_key(meds_file, diag_file)
    output_dir = f"tmp/pipeline_output_{cache_key}"
    
    # Expected pipeline output file needed for stats summarizer
    # Note: Stats summarizer only needs qc_flags.csv because it contains:
    #   - Drug classification info (drug_name, atc_code, drug_class, expected_icd10_codes,expected_icd10_ranges,actual_icd10_codes)
    #   - QC validation results (status, match_type, etc.)
    # The original medications and diagnoses files provide patient-level detail
    qc_flags = os.path.join(output_dir, "qc_flags.csv")

    print(f"Input files:")
    print(f"   - Medications: {meds_file}")
    print(f"   - Diagnoses: {diag_file}")
    print(f"\nWorking directory: {output_dir}")
    
    # Check if required pipeline output exists
    if not Path(qc_flags).exists():
        print(f"\nRequired pipeline output missing:")
        print(f"   - qc_flags.csv (contains drug classifications + QC results)")
        
        print("\nRunning 3-agent pipeline...")
        print("   1. Drug Identifier → extract drug names")
        print("   2. Drug Classifier → map to ATC codes → map to ICD-10 codes")
        print("   3. QC Evaluator → validate alignment (produces qc_flags.csv)\n")
        
        qc_flags = await run_pipeline(meds_file, diag_file, output_dir)
    else:
        print(f"\n[OK] Required pipeline output found:")
        print(f"   - QC Flags: {qc_flags}\n")
    
    # Create session with processed data
    # Note: Using original medications file (not drug_classifications) because we need
    # per-patient prescription data to count drugs per patient
    # qc_flags already contains the drug classification info (drug_name, atc_code, drug_class)
    print("Creating analysis session...")
    session_id = await create_stats_session(
        medications_file=meds_file,  # Original medications with patient-level data
        diagnoses_file=diag_file,
        qc_flags_file=qc_flags,
    )
    print(f"[OK] Session created: {session_id}\n")
    
    # Define demo queries for PROCESSED data analysis
    queries = [
        {
            "title": "Data Overview",
            "query": "What data has been loaded? Give me a quick summary with row counts and column names for each dataset."
        },
        {
            "title": "Drug Classification Summary",
            "query": "Show me a summary of the drug classifications from qc_flags_df. How many unique drugs were successfully classified vs unknown (check atc_code column)? Show ATC code distribution."
        },
        {
            "title": "Drugs Per Patient Histogram",
            "query": "Create a histogram showing the distribution of how many different drugs are prescribed per patient. Use medications_df (patient-level prescription data) to count unique drug descriptions per patient. Show both the histogram chart and a summary table showing: for each drug count (1 drug, 2 drugs, 3 drugs, etc.), how many patients have that many drugs prescribed."
        },
        {
            "title": "QC Analysis",
            "query": "Analyze the QC flags data. How many patients have QC issues? What are the most common types of issues?"
        },
        {
            "title": "Drug Class Patterns",
            "query": "Using qc_flags_df, group the classified drugs by drug_class (ATC anatomical group). Show me the top 5 drug classes and how many unique drugs fall into each."
        },
        {
            "title": "Follow-up: Success Rate",
            "query": "Based on the classification summary you showed earlier from qc_flags_df, calculate the overall success rate of drug classification. What percentage of unique drugs were successfully mapped to ATC codes (not UNKNOWN)?"
        }
    ]
    
    try:
        for i, item in enumerate(queries, 1):
            print("="*80)
            print(f"QUERY {i}: {item['title']}")
            print("="*80)
            print(f"Question: {item['query']}\n")
            
            response = await query_stats_session(session_id, item['query'])
            
            print("Response:")
            print("-"*80)
            print(response)
            print("\n")
            
            # Pause between queries
            if i < len(queries):
                await asyncio.sleep(2)
        
        print("="*80)
        print("[OK] DEMO COMPLETE")
        print("="*80)
        print("\nKey Takeaways:")
        print("1. The agent maintained context across all 5 queries")
        print("2. It could reference 'those top medications you showed earlier'")
        print("3. Code Execution enabled pandas analysis without pre-writing code")
        print("4. Multi-turn conversation felt natural and interactive")
        
    finally:
        # Clean up
        print(f"\nClosing session: {session_id}")
        await close_stats_session(session_id)
        print("[OK] Session closed")


if __name__ == "__main__":
    asyncio.run(main())
