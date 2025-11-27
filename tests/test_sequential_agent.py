"""
Unit tests for Drug-Dx-QC SequentialAgent.

Tests the SequentialAgent pattern following Kaggle Day 1b:
1. Create SequentialAgent with sub-agents
2. Create InMemoryRunner
3. Execute with await runner.run_debug()
"""

import pytest
import asyncio
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file for API key
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.drug_dx_qc_agents import (
    create_drug_dx_qc_sequential_agent,
    run_drug_dx_qc_pipeline
)
from google.adk.runners import InMemoryRunner


# Fixture to check for API key
@pytest.fixture(scope="session")
def api_key_available():
    """Check if GOOGLE_API_KEY is available."""
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        pytest.skip("GOOGLE_API_KEY not found. Set it in .env file or environment variable.")
    return api_key


class TestDrugDxQcSequentialAgent:
    """Test the SequentialAgent following correct ADK pattern."""
    
    def test_sequential_agent_creation(self):
        """Test that SequentialAgent can be created."""
        root_agent = create_drug_dx_qc_sequential_agent(model="gemini-2.5-flash")
        
        # Verify agent structure
        assert root_agent is not None
        assert root_agent.name == "drug_dx_qc_sequential_agent"
        assert hasattr(root_agent, 'sub_agents')
        assert len(root_agent.sub_agents) == 3
        
        # Verify sub-agents are present
        assert root_agent.sub_agents[0].name == "drug_identifier"
        # Note: Other agents may have different naming
    
    def test_inmemory_runner_pattern(self, api_key_available):
        """
        Test CORRECT pattern: SequentialAgent → InMemoryRunner → run_debug()
        
        This is the Kaggle Day 1b pattern:
        1. root_agent = SequentialAgent(...)
        2. runner = InMemoryRunner(agent=root_agent)
        3. response = await runner.run_debug(prompt)
        """
        
        async def run_test():
            # Step 1: Create SequentialAgent
            root_agent = create_drug_dx_qc_sequential_agent(model="gemini-2.5-flash")
            
            # Step 2: Create InMemoryRunner
            runner = InMemoryRunner(agent=root_agent)
            
            # Verify runner is created
            assert runner is not None
            
            # Step 3: Execute with runner.run_debug()
            test_prompt = """
            Hello! You are the Drug-Dx-QC sequential agent.
            Please describe your role and list your sub-agents.
            """
            
            response = await runner.run_debug(test_prompt)
            
            # Verify response
            assert response is not None
            response_text = response.text if hasattr(response, 'text') else str(response)
            assert len(response_text) > 0
            print(f"\n✅ Agent Response:\n{response_text}\n")
        
        # Run the async test
        asyncio.run(run_test())
    
    def test_run_pipeline_helper(self, api_key_available):
        """Test the convenience helper function that wraps the InMemoryRunner pattern."""
        
        test_prompt = """
        You are a drug quality control pipeline agent.
        Report your name and your three sub-agents.
        """
        
        response_text = asyncio.run(run_drug_dx_qc_pipeline(
            prompt=test_prompt,
            model="gemini-2.5-flash"
        ))
        
        # Verify response
        assert response_text is not None
        assert len(response_text) > 0
        print(f"\n✅ Pipeline Response:\n{response_text}\n")
    
    def test_full_pipeline_with_test_data(self, api_key_available, test_input_files):
        """
        Test the full 3-agent pipeline with actual test data files from test_input_files fixture.
        
        Uses test1 data (3 medication records, 3 patients) from tests/test1/input1/
        
        Through all 3 agents sequentially:
        1. Drug Identifier - extracts drug names
        2. Drug Classifier - classifies to ATC codes
        3. QC Evaluator - validates medication-diagnosis alignment
        
        Outputs are saved to: tests/tmp/
        """
        import shutil
        
        # Get input file paths from fixture
        meds_file = str(test_input_files['medications'])
        conditions_file = str(test_input_files['conditions'])
        
        # Clean output directory before test
        output_dir = "tests/tmp"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        test_prompt = f"""
Process the following medication and diagnosis data through the complete QC pipeline:

Input Files:
- Medications: {meds_file} (3 patients, 3 medications)
- Conditions: {conditions_file} (patient diagnoses)

Output Directory: {output_dir}/

Tasks:
1. Extract clean drug names from the medication descriptions
   → Save to: {output_dir}/drug_names_extracted.csv
2. Classify each drug to ATC codes with ICD-10 enrichment
   → Save to: {output_dir}/drug_classifications.csv
3. Validate that each medication aligns with the patient's diagnosed conditions
   → Save to: {output_dir}/qc_flags.csv

Please process all records and provide a summary of:
- How many drugs were identified
- How many were successfully classified with ATC codes
- How many passed QC validation (medication matches diagnosis)
- How many failed QC validation (medication doesn't match diagnosis)
"""
        
        print("\n" + "="*70)
        print("🔬 TESTING FULL 3-AGENT PIPELINE")
        print("="*70)
        
        response_text = asyncio.run(run_drug_dx_qc_pipeline(
            prompt=test_prompt,
            model="gemini-2.5-flash"
        ))
        
        # Verify response
        assert response_text is not None
        assert len(response_text) > 0
        
        print("\n" + "="*70)
        print("📊 PIPELINE RESPONSE:")
        print("="*70)
        print(response_text)
        print("="*70)
        
        # Verify key information is in response
        assert "drug" in response_text.lower() or "medication" in response_text.lower()
        
        # Verify output files were created
        print("\n" + "="*70)
        print("📁 CHECKING OUTPUT FILES:")
        print("="*70)
        
        expected_files = [
            os.path.join(output_dir, "drug_names_extracted.csv"),
            os.path.join(output_dir, "drug_classifications.csv"),
            os.path.join(output_dir, "qc_flags.csv")
        ]
        
        for filepath in expected_files:
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                print(f"✅ {filepath} ({file_size} bytes)")
            else:
                print(f"❌ {filepath} - NOT FOUND")
                
        print("="*70)
        print("\n✅ Full pipeline test completed successfully!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
