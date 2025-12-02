"""
Test Drug Identifier Agent using Python Orchestration (Production Pattern)

This test uses the Python Orchestration pattern where Python code controls
the workflow and calls the agent only for specific tasks (column detection).

This matches the production pattern in build_atc_database.py which calls
process_medications_file() directly.
"""

import pytest
import asyncio
import sys
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Load .env file for API key
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.drug_identifier import create_drug_identifier_agent, process_medications_file
from google.adk.runners import InMemoryRunner
from config import DEFAULT_MODEL
from tests.conftest import TEST_INPUT_FILES_TEST2, TEST_OUTPUT_FILES, BASELINE_FILES_TEST2


# Fixture to check for API key
@pytest.fixture(scope="session")
def api_key_available():
    """Check if GOOGLE_API_KEY is available."""
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        pytest.skip("GOOGLE_API_KEY not found. Set it in .env file or environment variable.")
    return api_key


class TestDrugIdentifierPythonOrchestration:
    """Test Drug Identifier using Python Orchestration (production pattern)."""
    
    def test_agent_creation_with_file_tools(self):
        """Verify agent has file I/O tools."""
        agent = create_drug_identifier_agent(model=DEFAULT_MODEL)
        
        # Verify agent exists
        assert agent is not None
        assert agent.name == "drug_identifier"
        
        # Verify tools are present
        assert hasattr(agent, 'tools')
        assert len(agent.tools) > 0
        
        # The agent should have 8 tools as defined in drug_identifier.py:
        # - get_csv_columns, get_csv_sample, get_column_data_sample
        # - extract_drug_name_regex, validate_drug_extraction
        # - read_csv_file, write_csv_file, write_dataframe_to_csv
        assert len(agent.tools) == 8, f"Expected 8 tools, got {len(agent.tools)}"
        
        print(f"✅ Agent created successfully")
        print(f"   Name: {agent.name}")
        print(f"   Model: {agent.model}")
        print(f"   Tools: {len(agent.tools)} tools configured")
        print(f"   Tools include: CSV I/O, drug extraction, validation")
    
    def test_extract_drugs_via_process_file(self, api_key_available):
        """
        Test drug extraction using Python Orchestration (production pattern).
        
        This test calls process_medications_file() directly, which:
        1. Uses agent to detect medication column (PHASE 1)
        2. Uses pure Python regex to extract drug names (PHASE 2)
        3. Uses pure Python validation (PHASE 3)
        
        This matches how build_atc_database.py works in production.
        
        This test generates: tests/tmp/drug_names_extracted_test2.csv
        """
        
        async def run_test():
            # Get input file from TEST_INPUT_FILES_TEST2
            meds_file = str(TEST_INPUT_FILES_TEST2['medications'])
            
            # Define output file using centralized config
            output_file = str(Path("tests/tmp/drug_names_extracted_test2.csv"))
            
            # Clean output directory
            output_dir = Path("tests/tmp")
            output_dir.mkdir(exist_ok=True)
            if Path(output_file).exists():
                Path(output_file).unlink()
            
            print("\n" + "="*70)
            print("🔬 TESTING PYTHON ORCHESTRATION PATTERN (Production)")
            print("="*70)
            print(f"Input:  {meds_file}")
            print(f"Output: {output_file}")
            print(f"Model:  {DEFAULT_MODEL}")
            print("="*70)
            
            # Call process_medications_file directly (same as build_atc_database.py)
            df_result = await process_medications_file(
                input_file=meds_file,
                output_file=output_file,
                error_log="tests/tmp/drug_identifier_test_errors.log",
                model=DEFAULT_MODEL,
                use_agent_validation=False  # Production uses fast Python validation
            )
            
            print("\n" + "="*70)
            print("📊 EXTRACTION COMPLETE:")
            print("="*70)
            print(f"   Rows extracted: {len(df_result)}")
            print(f"   Output file: {output_file}")
            print("="*70)
            
            return df_result, output_file
        
        # Run the async test
        df_result, output_file = asyncio.run(run_test())
        
        # Verify result
        assert df_result is not None
        assert len(df_result) > 0
        
        # Verify output file was created
        print("\n" + "="*70)
        print("📁 VERIFYING OUTPUT FILE:")
        print("="*70)
        
        assert os.path.exists(output_file), \
            f"Output file was not created: {output_file}"
        
        # Verify file contents
        df = pd.read_csv(output_file)
        
        print(f"✅ File created: {output_file}")
        print(f"   Rows: {len(df)}")
        print(f"   Columns: {', '.join(df.columns.tolist())}")
        
        # Verify structure
        assert 'drug_name' in df.columns, "Missing 'drug_name' column"
        assert 'drug_description' in df.columns, "Missing 'drug_description' column"
        
        # Verify we have data (test2 has 16 unique medications from 58 records)
        assert len(df) >= 10, f"Expected at least 10 rows, got {len(df)}"
        
        # Verify known drugs are present from test2 dataset
        drug_names = set(df['drug_name'].str.lower())
        print(f"\n   Extracted drugs: {', '.join(sorted(drug_names))}")
        
        assert 'amlodipine' in drug_names, "Missing 'amlodipine'"
        assert 'lisinopril' in drug_names, "Missing 'lisinopril'"
        assert 'fluticasone' in drug_names or 'fluticasone propionate' in drug_names, "Missing 'fluticasone'"
        
        print("\n✅ Agent Orchestration test passed!")
        print(f"   Agent successfully used file I/O tools to process {len(df)} medications")
    
    def test_compare_with_baseline(self):
        """
        Compare Python Orchestration output with baseline.
        
        This test compares the file generated by test_extract_drugs_via_process_file
        with the baseline drug names.
        
        Uses similarity scoring to account for minor extraction variations.
        """
        
        # File already generated by previous test
        output_file = str(Path("tests/tmp/drug_names_extracted_test2.csv"))
        baseline_file = str(BASELINE_FILES_TEST2['drug_names_extracted'])
        
        # Verify files exist
        assert os.path.exists(output_file), \
            f"Output not found: {output_file}\nRun test_extract_drugs_via_process_file first!"
        assert os.path.exists(baseline_file), \
            f"Baseline file not found: {baseline_file}"
        
        # Load results
        result_df = pd.read_csv(output_file)
        baseline_df = pd.read_csv(baseline_file)
        
        print("\n" + "="*70)
        print("📊 COMPARING OUTPUT WITH BASELINE:")
        print("="*70)
        print(f"Test output:  {len(result_df)} rows")
        print(f"Baseline:     {len(baseline_df)} rows")
        
        # Compare drug names (case-insensitive)
        result_drugs = set(result_df['drug_name'].str.lower())
        baseline_drugs = set(baseline_df['drug_name'].str.lower())
        
        print(f"\nTest drugs:     {', '.join(sorted(result_drugs))}")
        print(f"Baseline drugs: {', '.join(sorted(baseline_drugs))}")
        
        # Calculate similarity metrics
        matching_drugs = result_drugs & baseline_drugs
        missing_drugs = baseline_drugs - result_drugs
        extra_drugs = result_drugs - baseline_drugs
        
        # Similarity score (Jaccard similarity)
        union_drugs = result_drugs | baseline_drugs
        jaccard_score = len(matching_drugs) / len(union_drugs) if union_drugs else 0
        
        # Match rate (recall)
        match_rate = len(matching_drugs) / len(baseline_drugs) if baseline_drugs else 0
        
        print(f"\n📈 Similarity Metrics:")
        print(f"   ✅ Matching: {len(matching_drugs)}/{len(baseline_drugs)} drugs ({match_rate:.1%})")
        print(f"   📊 Jaccard Score: {jaccard_score:.1%}")
        
        if missing_drugs:
            print(f"   ⚠️  Missing from test output: {', '.join(sorted(missing_drugs))}")
        if extra_drugs:
            print(f"   ℹ️  Extra in test output: {', '.join(sorted(extra_drugs))}")
        
        # Assert reasonable similarity (agents are non-deterministic)
        # We accept 80% match rate as good enough given LLM variability
        assert match_rate >= 0.80, \
            f"Match rate too low: {match_rate:.1%} (expected >= 80%)"
        
        assert jaccard_score >= 0.70, \
            f"Jaccard score too low: {jaccard_score:.1%} (expected >= 70%)"
        
        print(f"\n✅ Comparison passed!")
        print(f"   Match Rate: {match_rate:.1%} (threshold: 80%)")
        print(f"   Jaccard Score: {jaccard_score:.1%} (threshold: 70%)")
        print(f"   Output is sufficiently similar to baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
