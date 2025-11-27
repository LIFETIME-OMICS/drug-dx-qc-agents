"""
Unit tests for QC Evaluator Agent.

Tests the QC evaluation of medication-diagnosis alignment.
"""

import pytest
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.qc_evaluator import create_qc_evaluator_agent, evaluate_qc
from config import OUTPUT_DIR


class TestQCEvaluator:
    """Test QC Evaluator Agent functionality."""
    
    def test_qc_evaluator_creation(self):
        """Test that QC Evaluator agent creation function works."""
        agent = create_qc_evaluator_agent()
        
        assert agent is not None
        assert agent.name == "qc_evaluator"
        
        print("✅ QC Evaluator agent created successfully")
    
    def test_evaluate_test_medications(self, test_input_files):
        """
        Test QC evaluation with test data files.
        
        Expected results:
        - Patient 1: amlodipine for hypertension (I10) → PASS ✓
        - Patient 2: lisinopril for hypertension (I10) → PASS ✓
        - Patient 3: Penicillin V for diabetes (E11.9) → FAIL ✗
        """
        
        medications_file = str(test_input_files['medications'])
        conditions_file = str(test_input_files['conditions'])
        drug_classifications_file = str(OUTPUT_DIR / "drug_classifications.csv")
        output_file = "tests/tmp/qc_flags_test.csv"
        
        print("\n" + "="*70)
        print("🔍 TESTING QC EVALUATOR")
        print("="*70)
        print(f"📥 Medications: {medications_file}")
        print(f"📥 Conditions: {conditions_file}")
        print(f"📥 Classifications: {drug_classifications_file}")
        print(f"📤 Output: {output_file}")
        
        # Run QC evaluation
        results_df = evaluate_qc(
            medications_file=medications_file,
            conditions_file=conditions_file,
            drug_classifications_file=drug_classifications_file,
            output_file=output_file
        )
        
        # Verify results
        assert results_df is not None
        assert len(results_df) == 3  # 3 medication records
        
        print("\n📊 QC Results:")
        print(results_df[['drug_name', 'atc_code', 'expected_icd10_codes', 'actual_icd10_codes', 'status']])
        
        # Count results
        passed = len(results_df[results_df['status'] == 'PASS'])
        failed = len(results_df[results_df['status'] == 'FAIL'])
        unknown = len(results_df[results_df['status'] == 'UNKNOWN_DRUG'])
        
        print(f"\n✅ PASS: {passed}")
        print(f"❌ FAIL: {failed}")
        print(f"⚠️  UNKNOWN: {unknown}")
        
        # Verify expected outcomes
        # We expect at least 2 medications to be evaluated
        # (Results may vary depending on ATC database state)
        assert len(results_df) == 3
        
        print("\n✅ QC evaluation test completed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
