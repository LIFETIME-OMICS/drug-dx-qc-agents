"""
Unit tests for the drug-dx-qc-agents pipeline with SequentialAgent.

Tests the three-agent pipeline with InMemoryRunner pattern:
1. drug_identifier extracts clean drug names (with column detection)
2. drug_classifier classifies drugs with hybrid WHO + LLM approach
3. qc_evaluator validates medication-diagnosis alignment

Run modes:
- Default: pytest tests\test_pipeline.py -v
  Uses session-scoped fixture with real WHO lookups (~30 seconds one-time cost)
  
- Mock mode: pytest tests\test_pipeline.py -v --mock
  Uses mocked WHO responses (instant, no API calls)

The session-scoped fixture runs the pipeline ONCE at test session start,
then all tests share the cached database. This reduces test time from 
9+ minutes (10 tests × 30 sec each) to ~30 seconds (one-time cost) or 
instant with --mock flag.
"""

import pytest
import pandas as pd
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.drug_dx_qc_agents import (
    create_drug_dx_qc_sequential_agent,
    run_drug_dx_qc_pipeline
)
from scripts.build_atc_database import build_atc_database


class TestBuildATCDatabasePipeline:
    """Test the complete three-agent pipeline with SequentialAgent and InMemoryRunner."""
    
    def test_sequential_agent_creation(self):
        """Test that SequentialAgent can be created."""
        # Create SequentialAgent (agent creation is in agents/ directory)
        pipeline = create_drug_dx_qc_sequential_agent(model="gemini-2.5-flash")
        
        assert pipeline is not None
        assert pipeline.name == "drug_dx_qc_sequential_agent"
        assert hasattr(pipeline, 'sub_agents')
        assert len(pipeline.sub_agents) == 3
    
    @pytest.mark.asyncio
    async def test_sequential_agent_execution_with_runner(self):
        """Test that SequentialAgent is executed with InMemoryRunner (Kaggle pattern)."""
        from google.adk.runners import InMemoryRunner
        
        # Step 1: Create SequentialAgent
        root_agent = create_drug_dx_qc_sequential_agent(model="gemini-2.5-flash")
        
        # Step 2: Create InMemoryRunner (THIS IS THE CORRECT PATTERN!)
        runner = InMemoryRunner(agent=root_agent)
        
        # Step 3: Execute with runner.run_debug()
        test_prompt = """
        You are a drug quality control pipeline.
        Report your capabilities and the sub-agents you coordinate.
        """
        
        response = await runner.run_debug(test_prompt)
        
        # Verify we got a response
        assert response is not None
        response_text = response.text if hasattr(response, 'text') else str(response)
        assert len(response_text) > 0
    
    def test_pipeline_with_test_medications(self, shared_test_database):
        """Test pipeline with medications_test.csv"""
        # Use shared database (already built)
        assert os.path.exists(shared_test_database['database_file'])
        assert os.path.exists(shared_test_database['classifications_file'])
        
        # Load and verify database
        with open(shared_test_database['database_file'], 'r') as f:
            db = json.load(f)
        
        assert len(db) == 3
        assert 'amlodipine' in db
        assert 'lisinopril' in db
        
    def test_drug_names_extracted(self, shared_test_database):
        """Test that drug names are correctly extracted."""
        # Use shared database
        extracted_file = shared_test_database['extracted_file']
        assert os.path.exists(extracted_file)
        
        df = pd.read_csv(extracted_file)
        assert len(df) == 3
        assert 'drug_name' in df.columns
        
        # Verify expected drugs are present
        drug_names = set(df['drug_name'].str.lower())
        assert 'amlodipine' in drug_names
        assert 'lisinopril' in drug_names
        assert 'penicillin v' in drug_names
        
    def test_drug_classifications_output(self, shared_test_database):
        """Test that drug classifications CSV is created correctly."""
        # Use shared database
        classifications_file = shared_test_database['classifications_file']
        assert os.path.exists(classifications_file)
        
        df = pd.read_csv(classifications_file)
        assert len(df) == 3
        
        # Verify required columns
        required_columns = ['drug_name', 'atc_code', 'atc_class', 'mapping_source', 'source']
        for col in required_columns:
            assert col in df.columns
            
    def test_atc_database_created(self, shared_test_database):
        """Test that atc_database.json is created."""
        # Use shared database
        db_file = shared_test_database['database_file']
        assert os.path.exists(db_file)
        
        with open(db_file, 'r') as f:
            db = json.load(f)
        
        assert len(db) == 3
        assert 'amlodipine' in db
        assert 'lisinopril' in db
        
    def test_known_drugs_classified_correctly(self, shared_test_database):
        """Test that known drugs get correct ATC codes."""
        # Use shared database
        with open(shared_test_database['database_file'], 'r') as f:
            db = json.load(f)
        
        # Verify known drug codes
        assert db['amlodipine']['code'] == 'C08CA01'
        assert db['amlodipine']['anatomical_group'] == 'C'
        
        assert db['lisinopril']['code'] == 'C09AA03'
        assert db['lisinopril']['anatomical_group'] == 'C'
        
    def test_mapping_sources_are_valid(self, shared_test_database):
        """Test that mapping_source field has valid values."""
        # Use shared database
        with open(shared_test_database['database_file'], 'r') as f:
            db = json.load(f)
        
        # Check valid mapping sources
        valid_sources = ['direct', 'pending_llm', 'llm_fallback']
        for drug, data in db.items():
            assert 'mapping_source' in data
            assert data['mapping_source'] in valid_sources
            
    def test_who_source_preferred(self, shared_test_database):
        """Test that WHO is the primary source for classifications."""
        # Use shared database
        df = pd.read_csv(shared_test_database['classifications_file'])
        
        # Count WHO-sourced drugs
        who_sourced = df[df['source'].str.contains('WHO', case=False, na=False)]
        
        # At least 2 out of 3 should be WHO-sourced
        assert len(who_sourced) >= 2
        
    def test_llm_fallback_flagged(self, shared_test_database):
        """Test that LLM fallback entries are flagged for verification."""
        # Use shared database
        with open(shared_test_database['database_file'], 'r') as f:
            db = json.load(f)
        
        # Check LLM fallback entries have needs_verification flag
        for drug, data in db.items():
            if data.get('mapping_source') == 'llm_fallback':
                assert data.get('needs_verification') == True
                assert 'LLM' in data.get('source', '')
                
    def test_log_files_created(self, shared_test_database):
        """Test that log files are created."""
        # Use shared database
        identifier_log = os.path.join(shared_test_database['log_dir'], 'drug_identifier_errors.log')
        classifier_log = os.path.join(shared_test_database['log_dir'], 'drug_classifier_errors.log')
        
        assert os.path.exists(identifier_log)
        assert os.path.exists(classifier_log)


    def test_output_matches_baseline(self, shared_test_database, baseline_files):
        """Test that pipeline output matches baseline for known test drugs."""
        # Use shared database
        output_df = pd.read_csv(shared_test_database['classifications_file'])
        
        # Load baseline
        baseline_file = baseline_files['drug_classifications']
        if not os.path.exists(baseline_file):
            pytest.skip(f"Baseline file not found: {baseline_file}")
        
        baseline_df = pd.read_csv(baseline_file)
        
        # Compare key fields (allow some flexibility for LLM variations)
        assert len(output_df) == len(baseline_df), "Different number of drugs"
        
        # Check each drug
        for _, baseline_row in baseline_df.iterrows():
            drug_name = baseline_row['drug_name']
            output_row = output_df[output_df['drug_name'] == drug_name]
            
            assert len(output_row) == 1, f"Drug not found in output: {drug_name}"
            output_row = output_row.iloc[0]
            
            # Critical fields must match exactly
            assert output_row['atc_code'] == baseline_row['atc_code'], \
                f"ATC code mismatch for {drug_name}: {output_row['atc_code']} vs {baseline_row['atc_code']}"
            
            # Mapping source: allow flexibility for WHO-sourced drugs
            # Both 'direct' and 'pending_llm' are acceptable for WHO lookups
            valid_who_sources = ['direct', 'pending_llm']
            if baseline_row['mapping_source'] in valid_who_sources:
                assert output_row['mapping_source'] in valid_who_sources, \
                    f"Expected WHO-sourced mapping for {drug_name}, got: {output_row['mapping_source']}"
                assert 'WHO' in output_row['source'], \
                    f"Expected WHO in source for {drug_name}"
            else:
                # For non-WHO sources, must match exactly
                assert output_row['mapping_source'] == baseline_row['mapping_source'], \
                    f"Mapping source mismatch for {drug_name}"


class TestPipelineRobustness:
    """Test pipeline handles edge cases and errors gracefully."""
    
    def test_duplicate_drugs_handled(self, tmp_path):
        """Test that duplicate drugs are handled correctly."""
        # Create CSV with duplicates using correct column name
        duplicate_file = tmp_path / "duplicates.csv"
        duplicate_file.write_text(
            "DESCRIPTION\n"
            "Amlodipine 5mg\n"
            "Amlodipine 10mg\n"
            "AMLODIPINE 5mg\n"
        )
        
        output_dir = tmp_path / "data"
        log_dir = tmp_path / "logs"
        output_dir.mkdir()
        log_dir.mkdir()
        
        # Run pipeline
        stats = build_atc_database(
            medications_file=str(duplicate_file),
            output_dir=str(output_dir),
            intermediate_dir=str(output_dir),
            log_dir=str(log_dir)
        )
        
        # Should process only unique drugs
        assert stats['total_drugs'] == 1
        
        # Check database has only one entry
        db_file = output_dir / 'atc_database.json'
        with open(db_file, 'r') as f:
            db = json.load(f)
        assert len(db) == 1
        assert 'amlodipine' in db


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
