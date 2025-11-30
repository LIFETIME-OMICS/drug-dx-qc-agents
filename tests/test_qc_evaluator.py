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
from conftest import (TEST2_MEDICATIONS_FILE, TEST2_CONDITIONS_FILE, TEST2_DRUG_CLASSIFICATIONS_FILE, 
                      TEST2_BASELINE_QC_FLAGS, TEST_QC_FLAGS_AGENT_ORCHESTRATION, 
                      DEFAULT_BATCH_SIZE, TEST_ROW_LIMIT)


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
        """Verify agent has file I/O tools including batch processing tools."""
        agent = create_qc_evaluator_agent(model="gemini-2.5-flash")
        
        # Verify agent exists
        assert agent is not None
        assert agent.name == "qc_evaluator"
        
        # Verify tools are present (should have 5 file I/O tools)
        assert hasattr(agent, 'tools')
        assert len(agent.tools) == 5, f"Expected 5 tools (read_csv, write_dataframe, append_row, get_csv_info, read_csv_batch), got {len(agent.tools)}"
        
        # Verify agent has instruction for independent QC evaluation
        assert hasattr(agent, 'instruction')
        instruction_lower = agent.instruction.lower()
        assert 'clinical qc evaluator' in instruction_lower
        assert 'independent quality control' in instruction_lower
        assert 'expert medical knowledge' in instruction_lower
        assert 'incremental' in instruction_lower, "Instruction should mention incremental writing"
        
        print(f"✅ Agent created with {len(agent.tools)} file I/O tools")
        print(f"   - read_csv_file, write_dataframe_to_csv, append_row_to_csv")
        print(f"   - get_csv_info, read_csv_batch (for incremental processing)")
        print(f"✅ Agent configured for independent medical QC evaluation with incremental writing")
    
    def test_qc_evaluation_via_prompt(self, api_key_available):
        """
        Test QC evaluation using Agent Orchestration pattern with test2 data.
        
        Uses 8-patient dataset with SNOMED CT codes that need to be mapped to ICD-10.
        The agent receives a natural language prompt with:
        - Input file paths (medications and conditions)
        - Output file path
        - Task description
        
        The agent uses independent medical knowledge (no classifications file needed).
        The LLM decides when to call file I/O tools and performs SNOMED CT to ICD-10 mapping.
        
        This test generates: tests/tmp/qc_flags_agent_orchestration.csv
        """
        
        async def run_test():
            # Use test2 data (8-patient dataset with SNOMED CT codes)
            medications_file = TEST2_MEDICATIONS_FILE
            conditions_file = TEST2_CONDITIONS_FILE
            
            # Verify inputs exist
            if not os.path.exists(medications_file):
                pytest.skip(f"Medications file not found: {medications_file}")
            if not os.path.exists(conditions_file):
                pytest.skip(f"Conditions file not found: {conditions_file}")
            
            # Define output file
            output_file = TEST_QC_FLAGS_AGENT_ORCHESTRATION
            
            # Clean output
            if os.path.exists(output_file):
                os.remove(output_file)
            
            # Create agent
            agent = create_qc_evaluator_agent(model="gemini-2.5-flash")
            
            # Create runner
            runner = InMemoryRunner(agent=agent)
            
            # Natural language prompt (Agent Orchestration)
            # Agent performs independent QC evaluation using its medical knowledge
            prompt = f"""
You are a clinical QC evaluator with independent medical expertise. Perform medication-diagnosis QC validation using your expert knowledge.

**Input Files:**
1. {medications_file} - patient medication records (contains patient, encounter, description)
2. {conditions_file} - patient conditions with SNOMED CT codes (patient, encounter, code, description)

**Important Notes:**
- The conditions file contains SNOMED CT codes paired with descriptions - use both code AND description to identify diagnoses
- You must map SNOMED CT to ICD-10 diagnosis codes using your medical coding expertise
- Use YOUR INDEPENDENT PHARMACOLOGY AND MEDICAL CODING KNOWLEDGE (no external classifications needed)

**Your Task:**
1. Read both CSV files using read_csv_file()

2. For EACH medication record:
   a) Extract the drug name from the description field
   b) Use your pharmacology knowledge to determine:
      - Appropriate ATC code using your drug classification expertise
      - Expected ICD-10 diagnosis codes for this medication's indications
      - Expected ICD-10 diagnosis ranges for the drug class
   
   c) For the patient's encounter, analyze the conditions data:
      - Read the SNOMED CT code AND description
      - Map to ICD-10 diagnosis codes using your medical coding expertise
      - Consider both the code value and the text description
   
   d) Evaluate medication-diagnosis alignment:
      - Compare expected ICD-10 codes with actual patient diagnoses
      - Determine if medication is appropriate for patient's condition
      - Status: PASS if diagnosis matches, FAIL if no match
      - Match type: "exact" for exact code match, "range" for category match, "none" for no match
   
   e) Provide clinical reasoning for your assessment

3. Write results to {output_file} using write_dataframe_to_csv() with these columns:
   patient_id, encounter_id, drug_name, drug_description, atc_code, drug_class,
   expected_icd10_codes, expected_icd10_ranges, actual_icd10_codes,
   status, match_type, matched_codes, reason

**Important:** 
- Process ONLY the medication records shown in the preview from read_csv_file() (typically 5 rows)
- DO NOT write Python code - use the tools directly to perform the evaluation
- Actually call write_dataframe_to_csv() to write the output file - don't just describe what you would do

Apply your deep medical knowledge of pharmacology, SNOMED CT to ICD-10 mapping, and clinical practice.

You MUST call write_dataframe_to_csv() tool with the results before completing this task.
"""
            
            print("\n" + "="*70)
            print("TESTING AGENT ORCHESTRATION PATTERN - QC EVALUATOR")
            print("="*70)
            print(f"Medications:     {medications_file}")
            print(f"Conditions:      {conditions_file}")
            print(f"Output:          {output_file}")
            print(f"Agent Mode:      Independent medical knowledge (no classifications file)")
            print("="*70)
            
            # Run agent with prompt and print progress
            print("\nRunning agent... (this may take 1-2 minutes)")
            print("   Agent will: read files -> load JSON -> match diagnoses -> write output\n")
            
            response = await runner.run_debug(prompt)
            
            # response is a list of events from run_debug()
            # Extract the final text response
            response_text = ""
            tool_calls = 0
            
            for event in response:
                # Check for tool calls
                if hasattr(event, 'content') and event.content:
                    for part in event.content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            tool_calls += 1
                            func_name = part.function_call.name if hasattr(part.function_call, 'name') else 'unknown'
                            print(f"   ✓ Tool called: {func_name}")
                        elif hasattr(part, 'text') and part.text:
                            response_text += part.text
            
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
        # Note: Agent processes limited rows due to read_csv_file tool returning only preview (first 5 rows)
        assert len(df) >= 3, f"Expected at least 3 rows processed by agent, got {len(df)}"
        print(f"\n⚠️  Note: Agent processed {len(df)} records (read_csv_file tool limitation)")
        print(f"   The read_csv_file tool shows only first 5 rows as preview")
        print(f"   Full dataset has 58 records, but agent can only see preview data")
        
        print("\n✅ Agent Orchestration test passed!")
        print(f"   Agent successfully evaluated {len(df)} medication-diagnosis pairs")
    
    def test_compare_with_baseline(self):
        """
        Compare Agent Orchestration output with test2 baseline.
        
        This test does NOT call the agent - it compares the file already
        generated by test_qc_evaluation_with_batch_processing with the baseline.
        
        Reorders the agent output to match medications file order before comparison,
        since agents may process rows out of order.
        
        Uses similarity scoring since agents are non-deterministic.
        """
        from agents.file_io_tools import reorder_csv_rows
        
        # File already generated by previous test
        output_file = TEST_QC_FLAGS_AGENT_ORCHESTRATION
        baseline_file = TEST2_BASELINE_QC_FLAGS
        medications_file = TEST2_MEDICATIONS_FILE
        
        # Verify files exist
        assert os.path.exists(output_file), \
            f"Agent output not found: {output_file}\nRun test_qc_evaluation_with_batch_processing first!"
        assert os.path.exists(baseline_file), \
            f"Baseline file not found: {baseline_file}"
        assert os.path.exists(medications_file), \
            f"Medications file not found: {medications_file}"
        
        print("\n" + "="*70)
        print("🔄 REORDERING AGENT OUTPUT TO MATCH MEDICATIONS FILE ORDER")
        print("="*70)
        
        # Reorder output to match medications file order (agent may process out of order)
        # Matches on patient + encounter + description to handle multiple meds per encounter
        result_df = reorder_csv_rows(
            input_file=output_file,
            reference_file=medications_file
        )
        print(f"✅ Reordered {len(result_df)} rows to match medications file order")
        
        # Load baseline and limit to TEST_ROW_LIMIT rows
        baseline_df = pd.read_csv(baseline_file)
        baseline_df = baseline_df.head(TEST_ROW_LIMIT)
        
        # Verify agent processed expected number of rows
        if len(result_df) < TEST_ROW_LIMIT:
            pytest.fail(
                f"Agent output incomplete: Expected {TEST_ROW_LIMIT} rows, got {len(result_df)} rows. "
                f"Agent may have counted header as data row or failed to process all medications."
            )
        
        print("\n" + "="*70)
        print("📊 COMPARING AGENT OUTPUT WITH BASELINE:")
        print("="*70)
        print(f"Agent output:  {len(result_df)} rows")
        print(f"Baseline:      {len(baseline_df)} rows (first {TEST_ROW_LIMIT} rows)")
        
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
    
    def test_qc_evaluation_with_batch_processing(self, api_key_available):
        """
        Test QC evaluation using incremental writing to handle all 58 medications.
        
        This test verifies the agent uses incremental writing (append_row_to_csv)
        to save progress after each medication evaluation.
        
        The agent should:
        1. Read medications data (any method: read_csv_file, read_csv_batch, etc.)
        2. Evaluate each medication one at a time (using independent medical knowledge)
        3. Call append_row_to_csv() immediately after each evaluation
        4. Print progress periodically
        
        Benefits of incremental writing:
        - Progress saved continuously (no data loss if interrupted)
        - Lower memory usage (no accumulation)
        - Real-time output file updates
        
        This test generates: tests/tmp/qc_flags_agent_orchestration.csv
        """
        
        async def run_test():
            # Use test2 data (8-patient dataset with SNOMED CT codes)
            medications_file = TEST2_MEDICATIONS_FILE
            conditions_file = TEST2_CONDITIONS_FILE
            
            # Verify inputs exist
            if not os.path.exists(medications_file):
                pytest.skip(f"Medications file not found: {medications_file}")
            if not os.path.exists(conditions_file):
                pytest.skip(f"Conditions file not found: {conditions_file}")
            
            # Define output file
            output_file = TEST_QC_FLAGS_AGENT_ORCHESTRATION
            
            # Clean output
            if os.path.exists(output_file):
                os.remove(output_file)
            
            # Create agent
            agent = create_qc_evaluator_agent(model="gemini-2.5-flash")
            
            # Create runner
            runner = InMemoryRunner(agent=agent)
            
            # Simplified prompt - focus on getting the task done
            # Agent uses independent medical knowledge - no classifications file needed
            prompt = f"""
You are a clinical QC evaluator with independent medical expertise. Evaluate medication-diagnosis alignment for {TEST_ROW_LIMIT} medications.

**Input Files:**
1. {medications_file} - medications (patient, encounter, description columns)
2. {conditions_file} - conditions with SNOMED CT codes

**Your Task:**
Read the first {TEST_ROW_LIMIT} medications and evaluate each one using YOUR MEDICAL KNOWLEDGE:

1. Read medications: use read_csv_batch(file_path="{medications_file}", start_row=0, batch_size={TEST_ROW_LIMIT})
2. Read conditions: use read_csv_file(file_path="{conditions_file}")

3. For EACH of the {TEST_ROW_LIMIT} medications:
   - Extract drug name from description
   - Determine ATC code using your pharmacology expertise
   - Map SNOMED CT to ICD-10 codes using your medical coding knowledge
   - Determine expected ICD-10 codes/ranges for the drug's indications
   - Check if patient has matching diagnosis
   - Call append_row_to_csv() to save this medication's result

**Output to:** {output_file}

**Required columns:** patient_id, encounter_id, drug_name, drug_description, atc_code, drug_class, expected_icd10_codes, expected_icd10_ranges, actual_icd10_codes, status, match_type, matched_codes, reason

**Column mapping:**
- medications `patient` -> output `patient_id`
- medications `encounter` -> output `encounter_id`
- medications `description` -> output `drug_description`

**Critical:** Use append_row_to_csv() for EACH medication - write one row at a time as you evaluate them.

Start now - read the data and evaluate all {TEST_ROW_LIMIT} medications.
"""
            
            print("\n" + "="*70)
            print("TESTING INCREMENTAL WRITING MODE - QC EVALUATOR")
            print("="*70)
            print(f"Medications:     {medications_file}")
            print(f"Conditions:      {conditions_file}")
            print(f"Output:          {output_file}")
            print(f"Row limit:       {TEST_ROW_LIMIT} (test mode)")
            print(f"Agent Mode:      Independent medical knowledge (no classifications file)")
            print(f"Expected append calls: {TEST_ROW_LIMIT}")
            print("="*70)
            
            # Run agent with prompt and print progress
            print("\nRunning agent with incremental writing... (this may take 2-3 minutes)")
            print("   Agent will: read data -> evaluate each medication -> append_row_to_csv\n")
            
            response = await runner.run_debug(prompt)
            
            # Extract response text and count tool calls
            response_text = ""
            tool_calls = 0
            append_calls = 0
            
            for event in response:
                if hasattr(event, 'content') and event.content and hasattr(event.content, 'parts') and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            tool_calls += 1
                            func_name = part.function_call.name if hasattr(part.function_call, 'name') else 'unknown'
                            if func_name == 'append_row_to_csv':
                                append_calls += 1
                            print(f"   Tool called: {func_name}")
                        elif hasattr(part, 'text') and part.text:
                            response_text += part.text
            
            print(f"\n   Total tool calls: {tool_calls}")
            print(f"   Append row calls: {append_calls}/{TEST_ROW_LIMIT}")
            
            print("\n" + "="*70)
            print("AGENT RESPONSE:")
            print("="*70)
            print(response_text)
            print("="*70)
            
            return response_text, output_file, append_calls
        
        # Run the async test
        response_text, output_file, append_calls = asyncio.run(run_test())
        
        # Verify response (text may be empty if agent only used tools)
        assert response_text is not None
        # Note: response_text may be empty if agent only made tool calls without final summary
        
        # Verify output file was created
        print("\n" + "="*70)
        print("VERIFYING INCREMENTAL WRITING OUTPUT:")
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
        assert 'status' in df.columns, "Missing 'status' column"
        
        # Verify incremental writing worked
        assert len(df) >= TEST_ROW_LIMIT * 0.9, \
            f"Expected around {TEST_ROW_LIMIT} rows with incremental writing, got {len(df)}"
        
        # Verify agent used incremental writing (append_row_to_csv)
        assert append_calls >= TEST_ROW_LIMIT * 0.9, \
            f"Expected {TEST_ROW_LIMIT} append_row_to_csv calls, got {append_calls}"
        
        print(f"\n✅ Incremental writing test passed!")
        print(f"   Agent processed {len(df)} medications")
        print(f"   Used append_row_to_csv {append_calls} times")
        print(f"   ✓ Incremental writing verified!")
        
        if response_text:
            print(f"\n📝 Agent Summary:")
            print(f"   {response_text[:200]}..." if len(response_text) > 200 else f"   {response_text}")
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
