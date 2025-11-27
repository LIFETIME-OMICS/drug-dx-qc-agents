"""
Test QC Evaluator Agent using Agent Orchestration (Prompt-Based)

This test uses the Agent Orchestration pattern where the LLM decides
when to call file I/O tools based on natural language prompts.

Contrasts with test_qc_evaluator.py which uses Python Orchestration
(calling evaluate_medications() function directly).
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

from agents.qc_evaluator import create_qc_evaluator_agent
from google.adk.runners import InMemoryRunner


# Fixture to check for API key
@pytest.fixture(scope="session")
def api_key_available():
    """Check if GOOGLE_API_KEY is available."""
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        pytest.skip("GOOGLE_API_KEY not found. Set it in .env file or environment variable.")
    return api_key


class TestQcEvaluatorAgentOrchestration:
    """Test QC Evaluator using Agent Orchestration (prompt-based file I/O)."""
    
    def test_agent_creation_with_file_tools(self):
        """Verify agent has file I/O tools."""
        agent = create_qc_evaluator_agent(model="gemini-2.5-flash")
        
        # Verify agent exists
        assert agent is not None
        assert agent.name == "qc_evaluator"
        
        # Verify tools are present
        assert hasattr(agent, 'tools')
        assert len(agent.tools) > 0
        
        # Check for file I/O tool names
        tool_names = []
        for tool in agent.tools:
            if hasattr(tool, 'function'):
                tool_names.append(tool.function.__name__)
        
        # Should have file I/O tools
        assert 'read_csv_file' in tool_names
        assert 'write_csv_file' in tool_names or 'write_dataframe_to_csv' in tool_names
        
        print(f"✅ Agent created with {len(agent.tools)} tools")
        print(f"   Tools: {', '.join(tool_names)}")
    
    def test_qc_evaluation_via_prompt(self, api_key_available, test_input_files, baseline_files):
        """
        Test QC evaluation using Agent Orchestration pattern.
        
        The agent receives a natural language prompt with:
        - Input file paths (classifications and conditions)
        - Output file path
        - Task description
        
        The LLM decides when to call file I/O tools.
        
        This test generates: tests/tmp/qc_flags_agent_orchestration.csv
        """
        
        async def run_test():
            # Use the outputs from previous steps
            # Note: drug_classifier generates drug_classifications CSV which we use here
            # (not the atc_database.json - that's only used by drug_classifier internally)
            medications_file = str(test_input_files['medications'])
            conditions_file = str(test_input_files['conditions'])
            classifications_file = "tests/tmp/drug_classifications_agent_orchestration.csv"
            
            # Verify inputs exist
            if not os.path.exists(medications_file):
                pytest.skip(f"Medications file not found: {medications_file}")
            if not os.path.exists(conditions_file):
                pytest.skip(f"Conditions file not found: {conditions_file}")
            if not os.path.exists(classifications_file):
                pytest.skip(f"Drug classifications not found: {classifications_file}\nRun test_drug_classifier2.py first!")
            
            # Define output file
            output_file = "tests/tmp/qc_flags_agent_orchestration.csv"
            
            # Clean output
            if os.path.exists(output_file):
                os.remove(output_file)
            
            # Create agent
            agent = create_qc_evaluator_agent(model="gemini-2.5-flash")
            
            # Create runner
            runner = InMemoryRunner(agent=agent)
            
            # Natural language prompt (Agent Orchestration)
            prompt = f"""
You are a clinical QC evaluator. Perform medication-diagnosis QC validation:

**Input Files:**
1. {medications_file} - contains PATIENT, ENCOUNTER, DESCRIPTION (medication descriptions)
2. {conditions_file} - contains patient, encounter, code (actual ICD-10 diagnoses)
3. {classifications_file} - CSV with drug classifications (drug_name, atc_code, atc_class, indication, icd10_codes)

**Task:**
1. Read all three CSV files using read_csv_file()
2. For each medication record:
   - Extract the clean drug name from the DESCRIPTION field (e.g., "amlodipine" from "amLODIPine 2.5 MG Oral Tablet")
   - Look up the drug in the classifications CSV to get:
     * atc_code
     * atc_class (drug_class)
     * icd10_codes (expected diagnoses - this is a JSON array string like '["I10", "I20.9"]')
   - Find the patient's actual diagnoses from the conditions file (match on patient + encounter)
   - Use check_diagnosis_match_tool() to validate if expected ICD-10 codes match actual diagnoses
   - Determine status: PASS if any match found, FAIL if no matches
   - Determine match_type: "exact" if exact code match, "partial" if range match, "none" if no match
   - List matched_codes: array of codes that matched
4. Create a results dataframe with these columns (in this order):
   - patient_id (from PATIENT column)
   - encounter_id (from ENCOUNTER column)
   - drug_name (cleaned/extracted name)
   - drug_description (original DESCRIPTION)
   - atc_code
   - drug_class
   - expected_icd10_codes (semicolon-separated)
   - expected_icd10_ranges (semicolon-separated)
   - actual_icd10_codes (semicolon-separated)
   - status (PASS or FAIL)
   - match_type (exact, partial, or none)
   - matched_codes (list format, e.g., "['I10']")
5. Write results to {output_file} using write_dataframe_to_csv()

Complete this task and confirm the file was written successfully.
"""
            
            print("\n" + "="*70)
            print("🔬 TESTING AGENT ORCHESTRATION PATTERN - QC EVALUATOR")
            print("="*70)
            print(f"Medications:     {medications_file}")
            print(f"Conditions:      {conditions_file}")
            print(f"Classifications: {classifications_file}")
            print(f"Output:          {output_file}")
            print("="*70)
            
            # Run agent with prompt and print progress
            print("\n🔄 Running agent... (this may take 1-2 minutes)")
            print("   Agent will: read files → load JSON → match diagnoses → write output\n")
            
            response = await runner.run_debug(prompt)
            
            # Extract response text
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            # Count tool calls to show progress
            tool_calls = 0
            for event in response.events:
                if hasattr(event, 'content') and event.content and hasattr(event.content, 'parts'):
                    for part in event.content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            tool_calls += 1
                            print(f"   ✓ Tool called: {part.function_call.name}")
            
            print(f"\n   Total tool calls: {tool_calls}")
            
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
        assert 'qc_status' in df.columns or 'status' in df.columns, "Missing QC status column"
        
        # Verify we have data
        assert len(df) >= 3, f"Expected at least 3 rows, got {len(df)}"
        
        print("\n✅ Agent Orchestration test passed!")
        print(f"   Agent successfully evaluated {len(df)} medication-diagnosis pairs")
    
    def test_compare_with_baseline(self, baseline_files):
        """
        Compare Agent Orchestration output with baseline.
        
        This test does NOT call the agent - it compares the file already
        generated by test_qc_evaluation_via_prompt with the baseline.
        
        Uses similarity scoring since agents are non-deterministic.
        """
        
        # File already generated by previous test
        output_file = "tests/tmp/qc_flags_agent_orchestration.csv"
        baseline_file = str(baseline_files['qc_flags_test'])
        
        # Verify files exist
        assert os.path.exists(output_file), \
            f"Agent output not found: {output_file}\nRun test_qc_evaluation_via_prompt first!"
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
        
        # Normalize status column name
        result_status_col = 'qc_status' if 'qc_status' in result_df.columns else 'status'
        baseline_status_col = 'qc_status' if 'qc_status' in baseline_df.columns else 'status'
        
        # Compare QC outcomes
        result_pass = len(result_df[result_df[result_status_col].str.upper() == 'PASS'])
        result_fail = len(result_df[result_df[result_status_col].str.upper() == 'FAIL'])
        
        baseline_pass = len(baseline_df[baseline_df[baseline_status_col].str.upper() == 'PASS'])
        baseline_fail = len(baseline_df[baseline_df[baseline_status_col].str.upper() == 'FAIL'])
        
        print(f"\nAgent QC Results:    {result_pass} PASS, {result_fail} FAIL")
        print(f"Baseline QC Results: {baseline_pass} PASS, {baseline_fail} FAIL")
        
        # Calculate similarity
        # Allow some tolerance since LLM may interpret QC rules differently
        pass_diff = abs(result_pass - baseline_pass)
        fail_diff = abs(result_fail - baseline_fail)
        total_diff = pass_diff + fail_diff
        
        similarity = 1.0 - (total_diff / len(baseline_df)) if len(baseline_df) > 0 else 0
        
        print(f"\n📈 Similarity Metrics:")
        print(f"   📊 QC Outcome Similarity: {similarity:.1%}")
        print(f"   ✅ PASS agreement: {min(result_pass, baseline_pass)}/{max(result_pass, baseline_pass)}")
        print(f"   ❌ FAIL agreement: {min(result_fail, baseline_fail)}/{max(result_fail, baseline_fail)}")
        
        # Assert reasonable similarity
        # Since QC evaluation involves clinical judgment, we accept 70% similarity
        assert similarity >= 0.70, \
            f"QC similarity too low: {similarity:.1%} (expected >= 70%)"
        
        print(f"\n✅ Comparison passed!")
        print(f"   QC Outcome Similarity: {similarity:.1%} (threshold: 70%)")
        print(f"   Agent QC evaluation is sufficiently similar to baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
