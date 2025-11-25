"""
Build ATC Database - Simple 2-Agent Pipeline

This script runs the 2-agent pipeline to build the ATC database:
1. Agent 1 (drug_identifier): Extract drug names from medications CSV
2. Agent 2 (drug_classifier): Classify drugs to ATC codes (WHO + LLM)

Usage:
    python scripts/build_atc_database2.py --medications data/medications.csv
    
Output:
    - output/atc_database.json: The final ATC database
    - output/drug_names_extracted.csv: Intermediate file from agent 1
    - output/drug_classifications.csv: Final classifications from agent 2
"""

import os
import sys
import argparse
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.drug_identifier import process_medications_file
from agents.drug_classifier import process_drug_names_file


def build_atc_database(
    medications_file: str,
    output_dir: str = "output"
) -> dict:
    """
    Build ATC database by running the 2-agent pipeline.
    
    Args:
        medications_file: Path to input medications CSV file
        output_dir: Directory for all outputs (default: output)
        
    Returns:
        Dictionary with statistics
    """
    print("\n" + "="*70)
    print("🏥 Building ATC Database - 2-Agent Pipeline")
    print("="*70)
    print(f"📥 Input: {medications_file}\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output files
    extracted_file = f"{output_dir}/drug_names_extracted.csv"
    classified_file = f"{output_dir}/drug_classifications.csv"
    identifier_log = f"{output_dir}/drug_identifier_errors.log"
    classifier_log = f"{output_dir}/drug_classifier_errors.log"
    
    # Step 1: Run drug_identifier agent
    print("\n" + "="*70)
    print("Step 1/2: Drug Identifier Agent")
    print("="*70)
    
    try:
        df_extracted = asyncio.run(process_medications_file(
            input_file=medications_file,
            output_file=extracted_file,
            error_log=identifier_log
        ))
        
        print(f"\n✅ Agent 1 complete: {len(df_extracted)} drugs extracted")
        print(f"📤 Output: {extracted_file}")
        
    except Exception as e:
        print(f"\n❌ Error in drug_identifier: {e}")
        raise
    
    # Step 2: Run drug_classifier agent
    print("\n" + "="*70)
    print("Step 2/2: Drug Classifier Agent (WHO + LLM)")
    print("="*70)
    
    try:
        df_classified = process_drug_names_file(
            input_file=extracted_file,
            output_file=classified_file,
            error_log=classifier_log
        )
        
        successful = len(df_classified[~df_classified['atc_code'].isin(['UNKNOWN', 'ERROR'])])
        
        print(f"\n✅ Agent 2 complete: {len(df_classified)} drugs classified")
        print(f"   - Successful: {successful}")
        print(f"   - Unknown/Error: {len(df_classified) - successful}")
        print(f"📤 Output: {classified_file}")
        
    except Exception as e:
        print(f"\n❌ Error in drug_classifier: {e}")
        raise
    
    # Summary
    print("\n" + "="*70)
    print("✅ ATC DATABASE BUILD COMPLETE")
    print("="*70)
    print(f"📁 Output files:")
    print(f"   - output/atc_database.json (ATC database production location)")
    print(f"   - {classified_file} (classifications)")
    print(f"   - {extracted_file} (extracted drug names)")
    print(f"\n🎉 Database ready for use!\n")
    
    return {
        'total_drugs': len(df_classified),
        'successful': successful,
        'extracted_file': extracted_file,
        'classified_file': classified_file
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build ATC database by running 2-agent pipeline (drug_identifier + drug_classifier)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/build_atc_database2.py --medications data/medications.csv
  python scripts/build_atc_database2.py --medications data/medications.csv --output-dir output
        """
    )
    
    parser.add_argument(
        "--medications",
        required=True,
        help="Path to medications CSV file"
    )
    
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for output files (default: output)"
    )
    
    args = parser.parse_args()
    
    try:
        stats = build_atc_database(
            medications_file=args.medications,
            output_dir=args.output_dir
        )
        
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        sys.exit(1)
