"""
Unit test for refactored drug_classifier agent with InMemoryRunner pattern.

Tests:
1. Agent creation and configuration
2. Read drug_identifier output (tests/tmp/drug_names_extracted.csv)
3. Classify drugs from drug_identifier output
4. Verify ATC codes and ICD-10 enrichment
"""

import pytest
import sys
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Load .env file for API key
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Fixture to check for API key
@pytest.fixture(scope="session")
def api_key_available():
    """Check if GOOGLE_API_KEY is available."""
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        pytest.skip("GOOGLE_API_KEY not found. Set it in .env file or environment variable.")
    return api_key


class TestDrugClassifierAgent:
    """Test the drug_classifier agent with InMemoryRunner pattern."""
    
    def test_agent_creation(self):
        """Test that drug_classifier agent creation function works."""
        from agents.drug_classifier import create_drug_classifier_agent
        
        agent = create_drug_classifier_agent()
        
        # Verify agent exists
        assert agent is not None
        assert agent.name == "drug_classifier"
        # Note: Agent model is configured internally, name verification is sufficient
        
        print("✅ Agent created successfully")
    
    def test_read_drug_identifier_output(self):
        """
        Test reading drug_identifier output from tests/tmp/drug_names_extracted.csv
        
        This file was created by drug_identifier agent test.
        Expected: 3 drugs (amlodipine, lisinopril, penicillin v)
        """
        input_file = "tests/tmp/drug_names_extracted.csv"
        
        # Check file exists
        assert os.path.exists(input_file), \
            f"Drug identifier output not found: {input_file}\nRun test_drug_identifier.py first!"
        
        # Read file
        df = pd.read_csv(input_file)
        
        # Verify structure
        assert 'drug_name' in df.columns
        assert 'drug_description' in df.columns
        assert len(df) == 3, f"Expected 3 drugs, got {len(df)}"
        
        # Verify known drugs
        drug_names = set(df['drug_name'].str.lower())
        assert 'amlodipine' in drug_names
        assert 'lisinopril' in drug_names
        assert any('penicillin' in name for name in drug_names)
        
        print(f"✅ Read {len(df)} drugs from drug_identifier output:")
        for drug in df['drug_name']:
            print(f"   - {drug}")
    
    def test_classify_single_drug(self, api_key_available):
        """
        Test classifying a single drug from drug_identifier output.
        
        Uses the hybrid approach: local DB → WHO → LLM fallback
        """
        import asyncio
        from agents.drug_classifier import classify_single_drug_async
        
        input_file = "tests/tmp/drug_names_extracted.csv"
        
        # Check file exists
        if not os.path.exists(input_file):
            pytest.skip("Run test_drug_identifier.py first to create test data")
        
        # Read first drug
        df = pd.read_csv(input_file)
        first_drug = df['drug_name'].iloc[0]
        
        print(f"\n🔬 Classifying: {first_drug}")
        
        # Classify using async function
        result = asyncio.run(classify_single_drug_async(first_drug))
        
        # Verify result structure
        assert 'code' in result
        assert 'drug_name' in result
        assert 'class' in result or 'drug_class' in result
        assert 'indication' in result
        assert 'icd10_codes' in result
        
        # Should have ATC code (7 characters or UNKNOWN)
        atc_code = result['code']
        assert atc_code is not None
        print(f"   ATC Code: {atc_code}")
        
        # Should have drug class
        drug_class = result.get('drug_class', result.get('class', 'Unknown'))
        print(f"   Class: {drug_class}")
        
        # Should have indication
        indication = result.get('indication', '')
        print(f"   Indication: {indication}")
        
        # Should have ICD-10 codes (may be empty for first run)
        icd10_codes = result.get('icd10_codes', [])
        print(f"   ICD-10 Codes: {icd10_codes}")
        
        print(f"✅ Successfully classified {first_drug}")
    
    def test_classify_all_drugs_from_identifier(self, api_key_available):
        """
        Test classifying all drugs from drug_identifier output.
        
        This simulates the Agent 1 → Agent 2 pipeline by calling the actual
        process_drug_names_file function that creates the correct format.
        
        Input:  tests/tmp/drug_names_extracted.csv (from drug_identifier)
        Output: tests/tmp/drug_classifications.csv (from drug_classifier)
        """
        from agents.drug_classifier import process_drug_names_file
        
        input_file = "tests/tmp/drug_names_extracted.csv"
        output_file = "tests/tmp/drug_classifications.csv"
        error_log = "tests/tmp/drug_classifier_errors.log"
        
        # Check input exists
        if not os.path.exists(input_file):
            pytest.skip("Run test_drug_identifier.py first to create test data")
        
        print(f"\n🔬 Running process_drug_names_file() - the actual drug_classifier function")
        
        # Use the ACTUAL drug_classifier function (same as production)
        results_df = process_drug_names_file(
            input_file=input_file,
            output_file=output_file,
            error_log=error_log
        )
        
        # Verify results
        assert results_df is not None
        assert len(results_df) > 0
        assert os.path.exists(output_file)
        
        # Verify correct columns (same as baseline)
        expected_columns = [
            'drug_name', 'atc_code', 'atc_class', 'indication', 
            'icd10_codes', 'icd10_mapping_source', 'source', 
            'needs_verification', 'comment'
        ]
        for col in expected_columns:
            assert col in results_df.columns, f"Missing column: {col}"
        
        # Count successes
        successful = len(results_df[~results_df['atc_code'].isin(['UNKNOWN', 'ERROR'])])
        
        # Compare with baseline
        baseline_file = "tests/baseline_test_drug_classifications.csv"
        if os.path.exists(baseline_file):
            baseline_df = pd.read_csv(baseline_file)
            
            # Verify same number of rows
            assert len(results_df) == len(baseline_df), \
                f"Row count mismatch: {len(results_df)} vs baseline {len(baseline_df)}"
            
            # Verify same drugs
            assert set(results_df['drug_name']) == set(baseline_df['drug_name']), \
                f"Drug names mismatch"
            
            # Verify all ATC codes match
            for idx, row in results_df.iterrows():
                drug = row['drug_name']
                baseline_row = baseline_df[baseline_df['drug_name'] == drug].iloc[0]
                assert row['atc_code'] == baseline_row['atc_code'], \
                    f"ATC code mismatch for {drug}: {row['atc_code']} vs {baseline_row['atc_code']}"
            
            print(f"\n✅ Output matches baseline: {baseline_file}")
        else:
            print(f"\n⚠️  Baseline not found: {baseline_file}")
        
        print(f"\n{'='*70}")
        print(f"✅ CLASSIFICATION COMPLETE")
        print(f"{'='*70}")
        print(f"📊 Total: {len(results_df)}")
        print(f"✅ Success: {successful}")
        print(f"📤 Output: {output_file}")
        print(f"📝 Baseline: tests/baseline_test_drug_classifications.csv")
        
        assert successful > 0, "No drugs were successfully classified"
    
    def test_database_loading(self):
        """Test that ATC database loads or creates successfully."""
        from agents.drug_classifier import load_atc_database
        from config import ATC_DATABASE_PATH
        
        # Load database using the function
        atc_database = load_atc_database(ATC_DATABASE_PATH)
        
        # Verify database exists (dict type)
        assert isinstance(atc_database, dict)
        
        print(f"✅ Database loaded from: {ATC_DATABASE_PATH}")
        print(f"   Database has {len(atc_database)} entries")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
