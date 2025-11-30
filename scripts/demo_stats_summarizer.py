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
from dotenv import load_dotenv

from config import TEST_MEDICATIONS_FILE, TEST_CONDITIONS_FILE, TEST_QC_FLAGS_FILE

load_dotenv()




async def main():
    """Run interactive demo of Stats Summarizer agent."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Interactive Stats Summarizer Demo')
    parser.add_argument('--medications', default=TEST_MEDICATIONS_FILE,
                        help='Path to medications CSV')
    parser.add_argument('--diagnoses', default=TEST_CONDITIONS_FILE,
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
    
    # Use OUTPUT_DIR from config.py for output directory
    from config import OUTPUT_DIR
    output_dir = str(OUTPUT_DIR)
    
    # Expected pipeline output file needed for stats summarizer
    # Use TEST_QC_FLAGS_FILE from config.py
    qc_flags = TEST_QC_FLAGS_FILE

    print(f"Input files:")
    print(f"   - Medications: {meds_file}")
    print(f"   - Diagnoses: {diag_file}")
    print(f"\nWorking directory: {output_dir}")
    
    # Check if required pipeline output exists
    if not Path(qc_flags).exists():
        print(f"\nRequired pipeline output missing:")
        print(f"   - qc_flags.csv (contains drug classifications + QC results)")
        print("\nPlease run the drug classification and QC pipeline first to generate qc_flags.csv.")
        return
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
