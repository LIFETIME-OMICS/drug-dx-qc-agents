"""
Unit test for refactored drug_identifier agent with InMemoryRunner pattern.

Tests:
1. Batch processing with the sync wrapper (covers agent execution)
2. Validation helper functions
3. Column detection tools
"""

import pytest
import sys
import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

# Load .env file for API key
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.drug_identifier import (
    process_medications_file_sync
)


# Fixture to check for API key
@pytest.fixture(scope="session")
def api_key_available():
    """Check if GOOGLE_API_KEY is available."""
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        pytest.skip("GOOGLE_API_KEY not found. Set it in .env file or environment variable.")
    return api_key


class TestDrugIdentifierAgent:
    """Test the refactored Drug Identifier Agent with InMemoryRunner via sync wrapper."""
    
    def test_batch_processing_sync(self, api_key_available):
        """
        Test full batch processing with sync wrapper.
        
        Input: tests/medications_test.csv (3 drugs)
        Expected output: 3 rows + header = 4 lines total
        Saves to: tests/tmp/drug_names_extracted.csv (for inspection)
        """
        input_file = "tests/medications_test.csv"
        output_dir = Path("tests/tmp")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "drug_names_extracted.csv"
        
        result_df = process_medications_file_sync(
            input_file=input_file,
            output_file=str(output_file),
            use_agent_validation=False  # Use fast path for testing
        )
        
        # Verify exactly 3 rows (from 3-drug test file)
        assert len(result_df) == 3, f"Expected 3 rows, got {len(result_df)}"
        
        # Verify required columns exist
        assert 'drug_description' in result_df.columns
        assert 'drug_name' in result_df.columns
        assert 'comment' in result_df.columns
        
        # Verify output file created
        assert output_file.exists()
        
        # Read output file and verify it has header + 3 rows = 4 lines
        with open(output_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 4, f"Expected 4 lines (header + 3 rows), got {len(lines)}"
        
        # Check for all 3 expected drugs from test file
        drug_names = set(result_df['drug_name'].str.lower())
        assert 'amlodipine' in drug_names, f"Missing amlodipine in {drug_names}"
        assert 'lisinopril' in drug_names, f"Missing lisinopril in {drug_names}"
        # Penicillin V is the correct extraction (not just "penicillin")
        assert any('penicillin' in name for name in drug_names), f"Missing penicillin in {drug_names}"
    
    def test_validation_function(self):
        """Test the validation helper function."""
        from agents.drug_identifier import validate_drug_extraction
        
        # Valid extraction
        result = validate_drug_extraction(
            "Amlodipine 5 MG Oral Tablet",
            "amlodipine"
        )
        assert result['is_valid'] == True
        assert len(result['issues']) == 0
        
        # Invalid: too short
        result = validate_drug_extraction(
            "Amlodipine 5 MG Oral Tablet",
            "am"
        )
        assert result['is_valid'] == False
        assert any("short" in issue.lower() for issue in result['issues'])
        
        # Invalid: contains formulation terms
        result = validate_drug_extraction(
            "Amlodipine 5 MG Oral Tablet",
            "amlodipine tablet"
        )
        assert result['is_valid'] == False
        assert any("dosage" in issue.lower() or "formulation" in issue.lower() 
                   for issue in result['issues'])
    
    def test_column_detection_tools(self):
        """Test the column detection helper tools."""
        from agents.drug_identifier import (
            get_csv_columns,
            get_csv_sample,
            get_column_data_sample
        )
        
        input_file = "tests/medications_test.csv"
        
        # Test get_csv_columns
        columns = get_csv_columns(input_file)
        assert isinstance(columns, list)
        assert len(columns) > 0
        assert 'DESCRIPTION' in columns or 'description' in columns
        
        # Test get_csv_sample
        sample = get_csv_sample(input_file, num_rows=2)
        assert isinstance(sample, str)
        assert len(sample) > 0
        
        # Test get_column_data_sample
        # First find a valid column name
        desc_col = 'DESCRIPTION' if 'DESCRIPTION' in columns else 'description'
        samples = get_column_data_sample(input_file, desc_col, num_samples=3)
        assert isinstance(samples, list)
        assert len(samples) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
