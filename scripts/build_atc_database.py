"""
Build ATC Database - Simple 2-Agent Pipeline

This script runs the 2-agent pipeline to build the ATC database:
1. Agent 1 (drug_identifier): Extract drug names from medications CSV
2. Agent 2 (drug_classifier): Classify drugs to ATC codes (WHO + LLM)

Usage:
    python scripts/build_atc_database.py --medications data/medications.csv
    
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
from config import DEFAULT_MODEL


def build_atc_database(
    medications_file: str,
    extracted_file: str = '',
    output_dir: str = "output",
    update_mode: str = "none"
) -> dict:
    """
    Build ATC database by running the 1 or 2-agent pipeline.
    
    Args:
        medications_file: Path to input medications CSV file
        extracted_file: Path to extracted drug names CSV (if provided, skips extraction)
        output_dir: Directory for all outputs (default: output)
        update_mode: How to handle existing database entries:
            - "none": Only add new drugs (default)
            - "add_unknown": Update entries with unknown fields
            - "always": Re-fetch all drugs
        
    Returns:
        Dictionary with statistics
    """
    print("\n" + "="*70)
    if extracted_file:
        print("🏥 Building ATC Database - 1-Agent Pipeline")
        print("="*70)
        print(f"📥 Input: {extracted_file}\n")        
    else:
        print("🏥 Building ATC Database - 2-Agent Pipeline")
        print("="*70)
        print(f"📥 Input: {medications_file}\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output files
    extracted_file_temp = extracted_file or f"{output_dir}/drug_names_extracted.csv"
    classified_file = f"{output_dir}/drug_classifications.csv"
    identifier_log = "logs/drug_identifier_errors.log"
    classifier_log = "logs/drug_classifier_errors.log"
    
    # Step 1: Run drug_identifier agent (only if no extracted file provided)
    if not extracted_file:
        print("\n" + "="*70)
        print("Step 1/2: Drug Identifier Agent")
        print("="*70)
        
        try:
            df_extracted = asyncio.run(process_medications_file(
                input_file=medications_file,
                output_file=extracted_file_temp,
                error_log=identifier_log,
                model=DEFAULT_MODEL
            ))
            
            print(f"\n✅ Agent 1 complete: {len(df_extracted)} drugs extracted")
            print(f"📤 Output: {extracted_file_temp}")
            
        except Exception as e:
            print(f"\n❌ Error in drug_identifier: {e}")
            raise
    
    # Step 2: Run drug_classifier agent
    print("\n" + "="*70)
    print("Step 2/2: Drug Classifier Agent (WHO + LLM)")
    print("="*70)
    
    try:
        df_classified = process_drug_names_file(
            input_file=extracted_file_temp,
            output_file=classified_file,
            error_log=classifier_log,
            model=DEFAULT_MODEL,
            update=update_mode
        )
        
        successful = len(df_classified[~df_classified['atc_code'].isin(['UNKNOWN', 'ERROR'])])
        
        print(f"\n✅ Agent 2 complete: {len(df_classified)} drugs classified")
        print(f"   - Successful: {successful}")
        print(f"   - Unknown/Error: {len(df_classified) - successful}")
        print(f"📤 Output: {classified_file}")
        
    except RuntimeError as e:
        if 'quota exhausted' in str(e).lower():
            print(f"\n🛑 Processing stopped: {e}")
            print(f"\n💡 Partial results saved to: {classified_file}")
            print("   You can resume processing later once quota resets.")
            sys.exit(2)  # Exit code 2 for quota exhausted
        raise
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
        'extracted_file': extracted_file_temp,
        'classified_file': classified_file
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build ATC database by running 2-agent pipeline (drug_identifier + drug_classifier)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (extract + classify):
  python scripts/build_atc_database.py --medications data/medications.csv
  
  # Classification only (debug mode - skips extraction):
  python scripts/build_atc_database.py --from-extracted output/drug_names_extracted.csv
        """
    )
    
    parser.add_argument(
        "--medications",
        help="Path to medications CSV file (full pipeline)"
    )
    
    parser.add_argument(
        "--from-extracted",
        help="Path to extracted drug names CSV (classification only - saves tokens)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for output files (default: output)"
    )
    
    parser.add_argument(
        "--update",
        choices=["none", "add_unknown", "always"],
        default="none",
        help="Database update mode: 'none' (add new only), 'add_unknown' (update incomplete), 'always' (re-fetch all)"
    )
    
    args = parser.parse_args()
    
    # Must specify either --medications or --from-extracted
    if not args.medications and not args.from_extracted:
        parser.error("Must specify either --medications or --from-extracted")
    
    if args.medications and args.from_extracted:
        parser.error("Cannot specify both --medications and --from-extracted")
    
    try:
        if args.from_extracted:
            # Debug mode: classification only
            stats = build_atc_database(
                medications_file='',
                extracted_file=args.from_extracted,
                output_dir=args.output_dir,
                update_mode=args.update
            )
        else:
            # Full pipeline
            stats = build_atc_database(
                medications_file=args.medications,
                extracted_file='',
                output_dir=args.output_dir,
                update_mode=args.update
            )
        
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        sys.exit(1)
