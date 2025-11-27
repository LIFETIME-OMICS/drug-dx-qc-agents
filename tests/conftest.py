"""
Pytest configuration for test suite.

Defines custom command-line options and shared fixtures.
"""

import pytest
import os
import json
from pathlib import Path
from unittest.mock import patch


def pytest_addoption(parser):
    """Add custom command line option for test modes."""
    parser.addoption(
        "--mock",
        action="store_true",
        default=False,
        help="Run tests with mocked WHO lookups (fastest)"
    )


# ============================================================================
# TEST FILE PATHS (Centralized)
# ============================================================================

TEST_DIR = Path(__file__).parent

# Test1: Original 3-patient test data with baselines
INPUT_DIR = TEST_DIR / "test1" / "input1"
BASELINE_DIR = TEST_DIR / "test1" / "baseline1"

# Test input files (test1)
TEST_INPUT_FILES = {
    'medications': INPUT_DIR / "medications_test.csv",
    'conditions': INPUT_DIR / "conditions_test.csv"
}

# Baseline files for test validation (test1)
BASELINE_FILES = {
    'drug_classifications': BASELINE_DIR / "baseline_test_drug_classifications.csv",
    'drug_names_extracted': BASELINE_DIR / "baseline_test_drug_names_extracted.csv",
    'atc_database': BASELINE_DIR / "baseline_atc_database.json",
    'test_atc_database': BASELINE_DIR / "baseline_test_atc_database.json",
    'qc_flags_test': BASELINE_DIR / "baseline_qc_flags_test.csv"
}

# Test2: 10-patient test data (no baselines - tests pipeline execution)
INPUT_DIR_TEST2 = TEST_DIR / "test2" / "input2"
TEST_INPUT_FILES_TEST2 = {
    'medications': INPUT_DIR_TEST2 / "medications_test.csv",
    'conditions': INPUT_DIR_TEST2 / "conditions_test.csv"
}


@pytest.fixture
def test_input_files():
    """Provide test1 input file paths to tests."""
    return TEST_INPUT_FILES


@pytest.fixture
def test_input_files_test2():
    """Provide test2 input file paths to tests (10 patients, no baselines)."""
    return TEST_INPUT_FILES_TEST2


@pytest.fixture
def baseline_files():
    """Provide baseline file paths to tests (test1 only)."""
    return BASELINE_FILES


@pytest.fixture(scope="session")
def use_mock_mode(request):
    """Determine if we should use mocked WHO lookups."""
    return request.config.getoption("--mock")


@pytest.fixture(scope="session")
def shared_test_database(tmp_path_factory, use_mock_mode):
    """
    Session-scoped fixture: Build database ONCE, share across all tests.
    
    This runs the pipeline once at test session start and creates a shared
    database that all tests can use, reducing test time from 9+ minutes to ~1 minute.
    
    Modes:
    - Default: Real WHO lookups (~30 seconds for 3 drugs)
    - --mock: Mocked WHO responses (instant)
    
    Returns:
        dict: Paths to generated files and metadata
    """
    import sys
    from pathlib import Path
    
    # Add project root to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.build_atc_database import build_atc_database
    
    # Create shared temp directory
    shared_tmp = tmp_path_factory.mktemp("shared_data")
    output_dir = shared_tmp / "data"
    log_dir = shared_tmp / "logs"
    output_dir.mkdir()
    log_dir.mkdir()
    
    if use_mock_mode:
        print("\n🚀 Using MOCKED WHO lookups (instant)")
        print(f"📋 Loading mock data from: {BASELINE_FILES['drug_classifications']}")
        
        # Load mock responses from baseline CSV
        import pandas as pd
        import json
        
        baseline_file = str(BASELINE_FILES['drug_classifications'])
        if not os.path.exists(baseline_file):
            raise FileNotFoundError(f"Baseline file not found: {baseline_file}")
        
        baseline_df = pd.read_csv(baseline_file)
        
        # Convert baseline to mock responses dictionary
        mock_responses = {}
        for _, row in baseline_df.iterrows():
            drug_name = row['drug_name'].lower()
            
            # Parse ICD-10 codes from string representation
            icd10_codes = []
            if pd.notna(row.get('icd10_codes')) and row['icd10_codes']:
                try:
                    icd10_codes = json.loads(row['icd10_codes'])
                except:
                    icd10_codes = []
            
            # Extract ATC code components
            atc_code = row['atc_code']
            mock_responses[drug_name] = {
                'code': atc_code,
                'drug_class': row['atc_class'],
                'anatomical_group': atc_code[0] if atc_code else None,
                'therapeutic_group': atc_code[:3] if len(atc_code) >= 3 else None,
                'pharmacological_group': atc_code[:4] if len(atc_code) >= 4 else None,
                'chemical_substance': atc_code if len(atc_code) >= 5 else None,
                'indication': row['indication'] if pd.notna(row['indication']) else '',
                'icd10_codes': icd10_codes,
                'indication_icd10_ranges': [],
                'icd10_mapping_source': row['icd10_mapping_source'] if 'icd10_mapping_source' in row else row.get('mapping_source', 'direct'),
                'source': row['source'] + ' (Mocked)',
                'needs_verification': bool(row['needs_verification'])
            }
            
            # Add original_name if it exists (for synonym lookups)
            if 'phenoxymethylpenicillin' in str(row.get('comment', '')).lower():
                mock_responses[drug_name]['original_name'] = 'phenoxymethylpenicillin'
        
        print(f"✅ Loaded {len(mock_responses)} mock drug responses from baseline")
        
        def mock_fetch_atc_from_who(drug_name, delay_seconds=0):
            """Mock WHO lookup - instant response from baseline CSV."""
            print(f"  🎭 MOCK: Fetching {drug_name} from baseline (no delay)")
            return mock_responses.get(drug_name.lower())
        
        # Apply mock
        with patch('agents.drug_classifier.fetch_atc_from_who', side_effect=mock_fetch_atc_from_who):
            # Run pipeline once with mocked WHO
            build_atc_database(
                medications_file=str(TEST_INPUT_FILES['medications']),
                output_dir=str(output_dir),
                intermediate_dir=str(output_dir),
                log_dir=str(log_dir)
            )
    else:
        print("\n🌐 Using REAL WHO lookups (~30 seconds for 3 drugs)")
        
        # Run pipeline once with real WHO lookups (session start, ~30 seconds)
        build_atc_database(
            medications_file=str(TEST_INPUT_FILES['medications']),
            output_dir=str(output_dir),
            intermediate_dir=str(output_dir),
            log_dir=str(log_dir)
        )
    
    # Return paths for tests to use
    return {
        'output_dir': str(output_dir),
        'log_dir': str(log_dir),
        'database_file': str(output_dir / 'atc_database.json'),
        'classifications_file': str(output_dir / 'drug_classifications.csv'),
        'extracted_file': str(output_dir / 'drug_names_extracted.csv'),
        'use_mock': use_mock_mode
    }
