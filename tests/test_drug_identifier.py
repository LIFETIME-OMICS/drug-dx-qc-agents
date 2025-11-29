"""
Test Drug Identifier Agent using Agent Orchestration (Prompt-Based)

This test uses the Agent Orchestration pattern where the LLM decides
when to call file I/O tools based on natural language prompts.

Contrasts with test_drug_identifier.py which uses Python Orchestration
(calling process_medications_file() function directly).
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

from agents.drug_identifier import create_drug_identifier_agent
from google.adk.runners import InMemoryRunner


# Fixture to check for API key
@pytest.fixture(scope="session")
def api_key_available():
    """Check if GOOGLE_API_KEY is available."""
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        pytest.skip("GOOGLE_API_KEY not found. Set it in .env file or environment variable.")
    return api_key


class TestDrugIdentifierAgentOrchestration:
    """Test Drug Identifier using Agent Orchestration (prompt-based file I/O)."""
    
    def test_agent_creation_with_file_tools(self):
        """Verify agent has file I/O tools."""
        agent = create_drug_identifier_agent(model="gemini-2.5-flash")
        
        # Verify agent exists
        assert agent is not None
        assert agent.name == "drug_identifier"
        
        # Verify tools are present (should include file I/O tools)
        assert hasattr(agent, 'tools')
        assert len(agent.tools) > 0
        
        # Check for file I/O tool names in the tool list
        tool_names = []
        for tool in agent.tools:
            if hasattr(tool, 'function'):
                tool_names.append(tool.function.__name__)
        
        # Should have file I/O tools
        assert 'read_csv_file' in tool_names
        assert 'write_csv_file' in tool_names or 'write_dataframe_to_csv' in tool_names
        
        print(f"✅ Agent created with {len(agent.tools)} tools")
        print(f"   Tools: {', '.join(tool_names)}")
    
    def test_extract_drugs_via_prompt(self, api_key_available, test_input_files):
        """
        Test drug extraction using Agent Orchestration pattern.
        
        The agent receives a natural language prompt with:
        - Input file path
        - Output file path
        - Task description
        
        The LLM decides when to call file I/O tools.
        
        This test generates: tests/tmp/drug_names_extracted_agent_orchestration.csv
        """
        
        async def run_test():
            # Get input file from fixture
            meds_file = str(test_input_files['medications'])
            
            # Define output file (used by next test for comparison)
            output_file = "tests/tmp/drug_names_extracted_agent_orchestration.csv"
            
            # Clean output directory
            output_dir = "tests/tmp"
            os.makedirs(output_dir, exist_ok=True)
            if os.path.exists(output_file):
                os.remove(output_file)
            
            # Create agent
            agent = create_drug_identifier_agent(model="gemini-2.5-flash")
            
            # Create runner
            runner = InMemoryRunner(agent=agent)
            
            # Natural language prompt (Agent Orchestration)
            prompt = f"""
You are a pharmaceutical data extraction expert. Please perform the following task:

**Input File:** {meds_file}

**Task:**
1. Read the CSV file using read_csv_file()
2. Identify which column contains medication descriptions
3. Extract clean drug names from each medication description
   - Remove dosages (e.g., "5 MG", "100mg")
   - Remove formulations (e.g., "Oral Tablet", "Injectable")
   - Remove routes (e.g., "Oral", "Intravenous")
   - Keep only the active pharmaceutical ingredient name
4. Create a result with these columns:
   - patient_id (from the original data)
   - encounter (from the original data) 
   - drug_description (original medication text)
   - drug_name (extracted clean name)
5. Write the results to: {output_file}
   Use write_dataframe_to_csv() to save the output

Please complete this task and confirm the file was written successfully.
"""
            
            print("\n" + "="*70)
            print("🔬 TESTING AGENT ORCHESTRATION PATTERN")
            print("="*70)
            print(f"Input:  {meds_file}")
            print(f"Output: {output_file}")
            print("="*70)
            
            # Run agent with prompt
            response = await runner.run_debug(prompt)
            
            # Extract response text
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            print("\n" + "="*70)
            print("📊 AGENT RESPONSE:")
            print("="*70)
            print(response_text)
            print("="*70)
            
            return response_text, output_file
        
        # Run the async test
        response_text, output_file = asyncio.run(run_test())
        
        # Verify response
        assert response_text is not None
        assert len(response_text) > 0
        
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
        
        # Verify we have data
        assert len(df) >= 3, f"Expected at least 3 rows, got {len(df)}"
        
        # Verify known drugs are present
        drug_names = set(df['drug_name'].str.lower())
        print(f"\n   Extracted drugs: {', '.join(drug_names)}")
        
        assert 'amlodipine' in drug_names, "Missing 'amlodipine'"
        assert 'lisinopril' in drug_names, "Missing 'lisinopril'"
        assert any('penicillin' in name for name in drug_names), "Missing 'penicillin'"
        
        print("\n✅ Agent Orchestration test passed!")
        print(f"   Agent successfully used file I/O tools to process {len(df)} medications")
    
    def test_compare_with_baseline(self, baseline_files):
        """
        Compare Agent Orchestration output with baseline.
        
        This test does NOT call the agent - it compares the file already
        generated by test_extract_drugs_via_prompt with the baseline.
        
        Uses similarity scoring since agents are non-deterministic.
        """
        
        # File already generated by previous test
        output_file = "tests/tmp/drug_names_extracted_agent_orchestration.csv"
        baseline_file = str(baseline_files['drug_names_extracted'])
        
        # Verify files exist
        assert os.path.exists(output_file), \
            f"Agent output not found: {output_file}\nRun test_extract_drugs_via_prompt first!"
        assert os.path.exists(baseline_file), \
            f"Baseline file not found: {baseline_file}"
        
        # Load results
        result_df = pd.read_csv(output_file)
        baseline_df = pd.read_csv(baseline_file)
        
        print("\n" + "="*70)
        print("📊 COMPARING AGENT OUTPUT WITH BASELINE:")
        print("="*70)
        print(f"Agent output:  {len(result_df)} rows")
        print(f"Baseline:      {len(baseline_df)} rows")
        
        # Compare drug names (case-insensitive)
        result_drugs = set(result_df['drug_name'].str.lower())
        baseline_drugs = set(baseline_df['drug_name'].str.lower())
        
        print(f"\nAgent drugs:    {', '.join(sorted(result_drugs))}")
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
            print(f"   ⚠️  Missing from agent output: {', '.join(sorted(missing_drugs))}")
        if extra_drugs:
            print(f"   ℹ️  Extra in agent output: {', '.join(sorted(extra_drugs))}")
        
        # Assert reasonable similarity (agents are non-deterministic)
        # We accept 80% match rate as good enough given LLM variability
        assert match_rate >= 0.80, \
            f"Match rate too low: {match_rate:.1%} (expected >= 80%)"
        
        assert jaccard_score >= 0.70, \
            f"Jaccard score too low: {jaccard_score:.1%} (expected >= 70%)"
        
        print(f"\n✅ Comparison passed!")
        print(f"   Match Rate: {match_rate:.1%} (threshold: 80%)")
        print(f"   Jaccard Score: {jaccard_score:.1%} (threshold: 70%)")
        print(f"   Agent output is sufficiently similar to baseline (non-deterministic LLM considered)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
