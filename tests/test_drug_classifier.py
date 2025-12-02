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
    create_drug_classifier_agent
)
from scripts.demo_qc_evaluator import evaluate_qc
from config import TMP_DIR, DEFAULT_MODEL
from tests.conftest import TEST_INPUT_FILES_TEST2, TEST_OUTPUT_FILES


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
    
    def test_drug_classification_multi_formulation(self, api_key_available, test_drug_params):
        """
        Test classify_drug() with multi-formulation support.
        
        Parametrized test that runs for multiple drugs:
        - amlodipine: 3 formulations (uses cache if available)
        - fluticasone: 6+ formulations (forces re-fetch from WHO)
        """
        
        drug_name = test_drug_params['drug_name']
        skip_cache = test_drug_params['skip_cache']
        min_formulations = test_drug_params['min_formulations']
        
        async def run_test():
            result = await classify_drug(drug_name, skip_cache=skip_cache)
            
            print("\n" + "="*70)
            print(f"Drug: {drug_name}")
            print(f"ATC Codes: {result['code']}")
            print(f"Drug Classes: {result['drug_class']}")
            print(f"Therapeutic Categories: {result['therapeutic_category']}")
            print(f"Anatomical Groups: {result['anatomical_group']}")
            print(f"Indication: {result['indication']}")
            print(f"ICD-10 Codes: {result['icd10_codes']}")
            print("="*70)
            
            return result
        
        result = asyncio.run(run_test())
        
        # Verify multi-formulation capture
        codes = result['code'].split('|')
        assert len(codes) >= min_formulations, \
            f"Expected at least {min_formulations} ATC codes for {drug_name}, got {len(codes)}: {codes}"
        
        # Verify pipe separators used for multi-formulation
        if len(codes) > 1:
            assert '|' in result['code'], f"Expected pipe separators in ATC codes for {drug_name}"
            assert '|' in result['drug_class'], f"Expected pipe separators in drug classes for {drug_name}"
        
        # Verify source and ICD-10 codes
        assert 'WHO' in result['source'], f"Expected WHO in source, got {result['source']}"
        assert len(result['icd10_codes']) >= 2, \
            f"Expected at least 2 ICD-10 codes for {drug_name}, got {len(result['icd10_codes'])}"
        
        # Special validation for fluticasone (respiratory + dermatological)
        if drug_name == 'fluticasone':
            assert any('R03' in code for code in codes), \
                f"Expected respiratory formulation (R03*) in {drug_name} codes: {codes}"
            assert any('D07' in code for code in codes), \
                f"Expected dermatological formulation (D07*) in {drug_name} codes: {codes}"
            
            icd_codes = result['icd10_codes']
            has_respiratory = any(code.startswith('J') for code in icd_codes)
            has_dermatology = any(code.startswith('L') for code in icd_codes)
            assert has_respiratory and has_dermatology, \
                f"Expected both respiratory (J*) and dermatological (L*) ICD-10 codes for {drug_name}"
        
        print(f"\n✅ {drug_name} classification passed - {len(codes)} formulations")
    
    def test_qc_evaluation(self, api_key_available):
        """
        Test QC evaluation with multi-formulation drug database.
        
        This tests the full pipeline: medications → conditions → QC evaluation
        using the updated ATC database with pipe-separated multi-formulation support.
        """
        
        print("\n" + "="*70)
        print("🔬 TESTING QC EVALUATION WITH MULTI-FORMULATION DATABASE")
        print("="*70)
        print(f"Using model: {DEFAULT_MODEL}")
        print("="*70)
        
        # Run QC evaluation (uses examples/atc_database.json with multi-formulation data)
        results = evaluate_qc()
        
        # Verify results structure
        assert results is not None, "QC evaluation returned None"
        
        print(f"\n✅ QC Evaluation completed")
        print(f"   Check output/qc_flags.csv for results")
    
    def test_compare_with_baseline(self, baseline_files_test2):
        """
        Compare drug classifications with baseline.
        
        This validates that the multi-formulation approach produces
        expected results by comparing against baseline drug classifications.
        """
        
        # Output from drug classification
        output_file = Path("output/drug_classifications.csv")
        baseline_file = baseline_files_test2['drug_classifications']
        
        # Verify files exist
        if not output_file.exists():
            pytest.skip("Run build_atc_database.py first to generate drug_classifications.csv")
        
        assert baseline_file.exists(), f"Baseline not found: {baseline_file}"
        
        # Load results
        result_df = pd.read_csv(str(output_file))
        baseline_df = pd.read_csv(str(baseline_file))
        
        print("\n" + "="*70)
        print("📊 COMPARING DRUG CLASSIFICATIONS WITH BASELINE")
        print("="*70)
        print(f"Current results: {len(result_df)} medications")
        print(f"Baseline:        {len(baseline_df)} medications")
        
        # Calculate classification rates
        current_classified = len(result_df[result_df['source'] != 'NOT_FOUND'])
        baseline_classified = len(baseline_df[baseline_df['source'] != 'NOT_FOUND'])
        
        current_rate = current_classified / len(result_df) if len(result_df) > 0 else 0
        baseline_rate = baseline_classified / len(baseline_df) if len(baseline_df) > 0 else 0
        
        print(f"\n📈 Classification Rates:")
        print(f"   Current:  {current_rate:.1%} ({current_classified}/{len(result_df)})")
        print(f"   Baseline: {baseline_rate:.1%} ({baseline_classified}/{len(baseline_df)})")
        
        # Compare all drugs' ICD-10 codes
        print(f"\n🔍 Comparing ICD-10 codes for all drugs:")
        mismatches = []
        for _, baseline_row in baseline_df.iterrows():
            drug_name = baseline_row['drug_name']
            current_row = result_df[result_df['drug_name'] == drug_name]
            
            if not current_row.empty:
                current_icd10 = current_row.iloc[0]['icd10_codes']
                baseline_icd10 = baseline_row['icd10_codes']
                
                # Handle NaN values (both should be NaN for NOT_FOUND drugs)
                current_is_nan = pd.isna(current_icd10)
                baseline_is_nan = pd.isna(baseline_icd10)
                
                # Only count as mismatch if one is NaN and the other isn't, or if both are non-NaN and different
                if current_is_nan != baseline_is_nan or (not current_is_nan and not baseline_is_nan and current_icd10 != baseline_icd10):
                    mismatches.append({
                        'drug': drug_name,
                        'current': current_icd10,
                        'baseline': baseline_icd10
                    })
        
        if mismatches:
            print(f"   ⚠️  Found {len(mismatches)} ICD-10 code mismatches:")
            for mismatch in mismatches:
                print(f"      {mismatch['drug']}:")
                print(f"        Current:  {mismatch['current']}")
                print(f"        Baseline: {mismatch['baseline']}")
            assert False, f"ICD-10 codes do not match baseline for {len(mismatches)} drugs"
        else:
            print(f"   ✅ All ICD-10 codes match baseline")
        
        # Compare fluticasone propionate specifically (multi-formulation drug)
        fluticasone_current = result_df[result_df['drug_name'].str.contains('fluticasone propionate', case=False, na=False)]
        fluticasone_baseline = baseline_df[baseline_df['drug_name'].str.contains('fluticasone propionate', case=False, na=False)]
        
        if not fluticasone_current.empty and not fluticasone_baseline.empty:
            current_data = fluticasone_current.iloc[0]
            baseline_data = fluticasone_baseline.iloc[0]
            
            print(f"\n🔬 Fluticasone Propionate (multi-formulation drug):")
            print(f"   Current ATC codes:  {current_data['atc_code']}")
            print(f"   Baseline ATC codes: {baseline_data['atc_code']}")
            
            # Count formulations (pipe-separated)
            current_codes = current_data['atc_code'].split('|') if pd.notna(current_data['atc_code']) else []
            baseline_codes = baseline_data['atc_code'].split('|') if pd.notna(baseline_data['atc_code']) else []
            
            print(f"   Current formulations:  {len(current_codes)}")
            print(f"   Baseline formulations: {len(baseline_codes)}")
            
            # Verify multi-formulation support
            assert len(current_codes) >= 3, f"Expected at least 3 formulations for fluticasone propionate, got {len(current_codes)}"
            
            # Check for respiratory codes (R03*)
            has_respiratory = any('R03' in code for code in current_codes)
            assert has_respiratory, f"Expected respiratory formulation (R03*) in fluticasone propionate codes: {current_codes}"
            
            # Check for dermatological codes (D07*)
            has_dermatological = any('D07' in code for code in current_codes)
            assert has_dermatological, f"Expected dermatological formulation (D07*) in fluticasone propionate codes: {current_codes}"
            
            # Compare ICD-10 codes
            current_icd10 = current_data['icd10_codes']
            baseline_icd10 = baseline_data['icd10_codes']
            print(f"   Current ICD-10 codes:  {current_icd10}")
            print(f"   Baseline ICD-10 codes: {baseline_icd10}")
            
            # Verify ICD-10 codes match
            assert current_icd10 == baseline_icd10, \
                f"ICD-10 codes mismatch for fluticasone propionate:\n  Current:  {current_icd10}\n  Baseline: {baseline_icd10}"
            
            print(f"   ✅ Multi-formulation support validated")
            print(f"   ✅ Contains respiratory (R03*) and dermatological (D07*) codes")
            print(f"   ✅ ICD-10 codes match baseline")
        
        print(f"\n✅ Baseline comparison passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
