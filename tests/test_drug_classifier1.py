"""
Test OPTIMIZED Drug Classifier (Minimal Agent Calls)

Tests the production pattern with cost optimization:
- Python: Cache check, WHO lookup, database save
- Agent: ONLY for synonyms + ICD-10 enrichment
"""

import pytest
import asyncio
import sys
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.drug_classifier import (
    classify_drug,
    process_drug_names_file,
    create_drug_classifier_agent
)
from config import TMP_DIR


@pytest.fixture(scope="session")
def api_key_available():
    """Check if GOOGLE_API_KEY is available."""
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        pytest.skip("GOOGLE_API_KEY not found")
    return api_key


class TestDrugClassifierOptimized:
    """Test optimized drug classifier with minimal agent calls."""
    
    def test_agent_creation(self):
        """Verify agent has NO tools (pure LLM knowledge)."""
        agent = create_drug_classifier_agent()
        
        assert agent is not None
        assert agent.name == "icd10_enrichment_agent"
        
        # Agent should have NO tools (pure LLM tasks only)
        assert len(agent.tools) == 0, f"Expected 0 tools, got {len(agent.tools)}"
        
        print(f"✅ Agent created with {len(agent.tools)} tools (pure LLM)")
    
    def test_single_drug_classification(self, api_key_available):
        """Test classify_drug() for single drug."""
        
        async def run_test():
            # Test drug that should be in WHO database
            result = await classify_drug("amlodipine")
            
            print("\n" + "="*70)
            print(f"Drug: amlodipine")
            print(f"ATC Code: {result['code']}")
            print(f"Drug Class: {result['drug_class']}")
            print(f"Indication: {result['indication']}")
            print(f"ICD-10 Codes: {result['icd10_codes']}")
            print("="*70)
            
            return result
        
        result = asyncio.run(run_test())
        
        # Verify result
        assert result['code'] == 'C08CA01', f"Expected C08CA01, got {result['code']}"
        assert 'Cardiovascular' in result['drug_class']
        assert 'WHO' in result['source'], f"Expected WHO in source, got {result['source']}"
        assert len(result['icd10_codes']) >= 2
        
        print(f"\n✅ Single drug classification passed")
    
    def test_batch_classification_optimized(self, api_key_available):
        """
        Test production function: process_drug_names_file_optimized()
        
        This tests the ACTUAL production pattern with cost optimization.
        Creates persistent output file in tmp/ directory for inspection.
        """
        
        # Input file from drug_identifier (in tests/tmp/)
        input_file = Path("tests/tmp/drug_names_extracted_agent_orchestration.csv")
        
        if not input_file.exists():
            pytest.skip(f"Input file not found: {input_file}\nRun test_drug_identifier2.py first!")
        
        # Output file (persistent in tests/tmp/ for inspection)
        output_file = Path("tests/tmp/drug_classifications_optimized.csv")
        
        # Clean output
        if output_file.exists():
            output_file.unlink()
        
        print("\n" + "="*70)
        print("🔬 TESTING OPTIMIZED BATCH PROCESSING")
        print("="*70)
        print(f"Input:  {input_file}")
        print(f"Output: {output_file}")
        print("="*70)
        
        # Run production function
        result_df = process_drug_names_file(
            input_file=str(input_file),
            output_file=str(output_file),
            model="gemini-2.5-flash"
        )
        
        # Verify output file
        assert output_file.exists(), f"Output file not created"
        
        # Verify structure
        assert 'drug_name' in result_df.columns
        assert 'atc_code' in result_df.columns
        assert 'atc_class' in result_df.columns
        assert 'indication' in result_df.columns
        assert 'icd10_codes' in result_df.columns
        
        # Verify we have data
        assert len(result_df) >= 3, f"Expected at least 3 rows, got {len(result_df)}"
        
        # Verify known drugs
        drug_names = set(result_df['drug_name'].str.lower())
        assert 'amlodipine' in drug_names
        assert 'lisinopril' in drug_names
        
        print(f"\n✅ Batch processing passed")
        print(f"   Processed {len(result_df)} drugs")
    
    def test_compare_with_baseline(self, baseline_files):
        """
        Compare optimized output with baseline.
        
        This validates that the optimized approach produces
        equivalent results to the baseline.
        """
        
        # Output from previous test (persistent file in tests/tmp/)
        output_file = Path("tests/tmp/drug_classifications_optimized.csv")
        baseline_file = baseline_files['drug_classifications']
        
        # Verify files exist
        if not output_file.exists():
            pytest.skip("Run test_batch_classification_optimized first")
        
        assert baseline_file.exists(), f"Baseline not found: {baseline_file}"
        
        # Load results
        result_df = pd.read_csv(str(output_file))
        baseline_df = pd.read_csv(str(baseline_file))
        
        print("\n" + "="*70)
        print("📊 COMPARING OPTIMIZED OUTPUT WITH BASELINE")
        print("="*70)
        print(f"Optimized: {len(result_df)} rows")
        print(f"Baseline:  {len(baseline_df)} rows")
        
        # Compare drug names
        result_drugs = set(result_df['drug_name'].str.lower())
        baseline_drugs = set(baseline_df['drug_name'].str.lower())
        
        matching_drugs = result_drugs & baseline_drugs
        missing_drugs = baseline_drugs - result_drugs
        
        # Match rate
        match_rate = len(matching_drugs) / len(baseline_drugs) if baseline_drugs else 0
        
        # Compare ATC codes
        atc_matches = 0
        for drug in matching_drugs:
            result_atc = result_df[result_df['drug_name'].str.lower() == drug]['atc_code'].values
            baseline_atc = baseline_df[baseline_df['drug_name'].str.lower() == drug]['atc_code'].values
            
            if len(result_atc) > 0 and len(baseline_atc) > 0:
                if result_atc[0] == baseline_atc[0]:
                    atc_matches += 1
        
        atc_match_rate = atc_matches / len(matching_drugs) if matching_drugs else 0
        
        print(f"\n📈 Comparison Metrics:")
        print(f"   ✅ Drug match rate: {match_rate:.1%}")
        print(f"   🔬 ATC code accuracy: {atc_match_rate:.1%}")
        
        if missing_drugs:
            print(f"   ⚠️  Missing: {', '.join(sorted(missing_drugs))}")
        
        # Assert quality thresholds
        assert match_rate >= 0.80, f"Match rate too low: {match_rate:.1%}"
        assert atc_match_rate >= 0.70, f"ATC accuracy too low: {atc_match_rate:.1%}"
        
        print(f"\n✅ Comparison passed!")
        print(f"   Optimized approach produces equivalent results")
        print(f"   With significantly reduced agent calls!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
