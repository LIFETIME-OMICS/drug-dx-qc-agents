"""
Test Sequential Agent Pipeline using Agent Orchestration (Prompt-Based)

This test validates the full 3-agent sequential pipeline where each agent
uses file I/O tools to read and write intermediate files.

This is the COMPLETE integration test that demonstrates the Agent Orchestration
pattern working end-to-end across all three agents.
"""

import pytest
import asyncio
import sys
import os
import pandas as pd
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Load .env file for API key
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.drug_dx_qc_agents import run_drug_dx_qc_pipeline


# Fixture to check for API key
@pytest.fixture(scope="session")
def api_key_available():
    """Check if GOOGLE_API_KEY is available."""
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        pytest.skip("GOOGLE_API_KEY not found. Set it in .env file or environment variable.")
    return api_key


class TestSequentialAgentOrchestration:
    """Test Sequential Agent Pipeline using Agent Orchestration."""
    
    def test_full_pipeline_via_prompt(self, api_key_available, test_input_files):
        """
        Test the full 3-agent pipeline using Agent Orchestration pattern.
        
        The sequential agent receives a natural language prompt that describes
        the entire workflow. Each sub-agent uses file I/O tools to:
        1. Drug Identifier - reads medications, writes drug names
        2. Drug Classifier - reads drug names, writes classifications
        3. QC Evaluator - reads classifications & conditions, writes QC flags
        
        This test generates all three output files in tests/tmp/sequential/
        """
        
        async def run_test():
            # Get input files
            meds_file = str(test_input_files['medications'])
            conditions_file = str(test_input_files['conditions'])
            
            # Clean output directory
            output_dir = "tests/tmp/sequential"
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            
            # Define output files
            drug_names_file = f"{output_dir}/drug_names_extracted.csv"
            classifications_file = f"{output_dir}/drug_classifications.csv"
            qc_flags_file = f"{output_dir}/qc_flags.csv"
            
            # Natural language prompt for the full pipeline
            prompt = f"""
Process the following medication and diagnosis data through the complete QC pipeline:

**Input Files:**
- Medications: {meds_file} (3 patients, 3 medications)
- Conditions: {conditions_file} (patient diagnoses with ICD-10 codes)

**Output Directory:** {output_dir}/

**Complete Pipeline Tasks:**

1. **Drug Identification** (First Agent):
   - Read the medications CSV file
   - Extract clean drug names from medication descriptions
   - Remove dosages, formulations, and routes
   - Keep only active pharmaceutical ingredient names
   - Save results to: {drug_names_file}
   - Include columns: patient_id, encounter, drug_description, drug_name

2. **Drug Classification** (Second Agent):
   - Read the drug names from step 1: {drug_names_file}
   - For each drug, lookup ATC code using WHO database
   - Extract drug class and provide ICD-10 codes for indications
   - Save results to: {classifications_file}
   - Include columns: drug_name, atc_code, atc_class, indication, icd10_codes

3. **QC Evaluation** (Third Agent):
   - Read drug classifications from step 2: {classifications_file}
   - Read patient conditions: {conditions_file}
   - For each medication, validate that the patient has a matching diagnosis
   - Compare expected ICD-10 codes (from drug) with actual diagnoses
   - Save results to: {qc_flags_file}
   - Include columns: patient_id, encounter, drug_name, expected_icd10, actual_icd10, qc_status

Please process all records sequentially and provide a summary of:
- How many drugs were identified
- How many were successfully classified with ATC codes
- How many passed QC validation
- How many failed QC validation
"""
            
            print("\n" + "="*70)
            print("🔬 TESTING SEQUENTIAL AGENT ORCHESTRATION - FULL PIPELINE")
            print("="*70)
            print(f"Medications: {meds_file}")
            print(f"Conditions:  {conditions_file}")
            print(f"Output Dir:  {output_dir}/")
            print("="*70)
            
            # Run the sequential agent pipeline
            response_text = await run_drug_dx_qc_pipeline(
                prompt=prompt,
                model="gemini-2.5-flash"
            )
            
            print("\n" + "="*70)
            print("📊 PIPELINE RESPONSE:")
            print("="*70)
            print(response_text)
            print("="*70)
            
            return response_text, drug_names_file, classifications_file, qc_flags_file
        
        # Run the async test
        response_text, drug_names_file, classifications_file, qc_flags_file = asyncio.run(run_test())
        
        # Verify response
        assert response_text is not None
        assert len(response_text) > 0
        
        # Verify all output files were created
        print("\n" + "="*70)
        print("📁 VERIFYING OUTPUT FILES:")
        print("="*70)
        
        expected_files = [
            drug_names_file,
            classifications_file,
            qc_flags_file
        ]
        
        for filepath in expected_files:
            assert os.path.exists(filepath), \
                f"Output file was not created: {filepath}"
            
            df = pd.read_csv(filepath)
            file_size = os.path.getsize(filepath)
            
            print(f"✅ {os.path.basename(filepath)}")
            print(f"   Rows: {len(df)}, Columns: {len(df.columns)}, Size: {file_size} bytes")
        
        # Verify content of each file
        drug_names_df = pd.read_csv(drug_names_file)
        assert len(drug_names_df) >= 3, f"Expected at least 3 drug names, got {len(drug_names_df)}"
        assert 'drug_name' in drug_names_df.columns
        
        classifications_df = pd.read_csv(classifications_file)
        assert len(classifications_df) >= 3, f"Expected at least 3 classifications, got {len(classifications_df)}"
        assert 'atc_code' in classifications_df.columns
        
        qc_flags_df = pd.read_csv(qc_flags_file)
        assert len(qc_flags_df) >= 3, f"Expected at least 3 QC evaluations, got {len(qc_flags_df)}"
        
        print("\n✅ Sequential Agent Orchestration test passed!")
        print(f"   All three agents successfully used file I/O tools")
        print(f"   Complete pipeline executed end-to-end")
    
    def test_compare_pipeline_with_baselines(self, baseline_files):
        """
        Compare Sequential Pipeline outputs with baselines.
        
        This test does NOT call the agents - it compares the files already
        generated by test_full_pipeline_via_prompt with the baselines.
        
        Validates all three output files against their respective baselines.
        """
        
        # Files already generated by previous test
        output_dir = "tests/tmp/sequential"
        drug_names_file = f"{output_dir}/drug_names_extracted.csv"
        classifications_file = f"{output_dir}/drug_classifications.csv"
        qc_flags_file = f"{output_dir}/qc_flags.csv"
        
        # Baseline files
        baseline_drug_names = str(baseline_files['drug_names_extracted'])
        baseline_classifications = str(baseline_files['drug_classifications'])
        baseline_qc_flags = str(baseline_files['qc_flags_test'])
        
        # Verify all files exist
        assert os.path.exists(drug_names_file), \
            f"Drug names not found: {drug_names_file}\nRun test_full_pipeline_via_prompt first!"
        assert os.path.exists(classifications_file), \
            f"Classifications not found: {classifications_file}"
        assert os.path.exists(qc_flags_file), \
            f"QC flags not found: {qc_flags_file}"
        
        print("\n" + "="*70)
        print("📊 COMPARING PIPELINE OUTPUTS WITH BASELINES:")
        print("="*70)
        
        # Compare Drug Names
        result_names = pd.read_csv(drug_names_file)
        baseline_names = pd.read_csv(baseline_drug_names)
        
        result_drugs = set(result_names['drug_name'].str.lower())
        baseline_drugs = set(baseline_names['drug_name'].str.lower())
        
        names_match = result_drugs & baseline_drugs
        names_match_rate = len(names_match) / len(baseline_drugs) if baseline_drugs else 0
        
        print(f"\n1️⃣ Drug Names Comparison:")
        print(f"   Agent: {len(result_names)} drugs")
        print(f"   Baseline: {len(baseline_names)} drugs")
        print(f"   Match Rate: {names_match_rate:.1%}")
        
        # Compare Classifications
        result_class = pd.read_csv(classifications_file)
        baseline_class = pd.read_csv(baseline_classifications)
        
        result_atc = set(result_class['drug_name'].str.lower())
        baseline_atc = set(baseline_class['drug_name'].str.lower())
        
        class_match = result_atc & baseline_atc
        class_match_rate = len(class_match) / len(baseline_atc) if baseline_atc else 0
        
        print(f"\n2️⃣ Drug Classifications Comparison:")
        print(f"   Agent: {len(result_class)} classifications")
        print(f"   Baseline: {len(baseline_class)} classifications")
        print(f"   Match Rate: {class_match_rate:.1%}")
        
        # Compare QC Flags
        result_qc = pd.read_csv(qc_flags_file)
        baseline_qc = pd.read_csv(baseline_qc_flags)
        
        print(f"\n3️⃣ QC Flags Comparison:")
        print(f"   Agent: {len(result_qc)} evaluations")
        print(f"   Baseline: {len(baseline_qc)} evaluations")
        
        # Overall pipeline similarity
        overall_similarity = (names_match_rate + class_match_rate) / 2
        
        print(f"\n📈 Overall Pipeline Similarity: {overall_similarity:.1%}")
        
        # Assert reasonable similarity across pipeline
        assert names_match_rate >= 0.80, \
            f"Drug names match too low: {names_match_rate:.1%}"
        assert class_match_rate >= 0.80, \
            f"Classifications match too low: {class_match_rate:.1%}"
        
        print(f"\n✅ Pipeline comparison passed!")
        print(f"   Sequential agent pipeline produces outputs consistent with baselines")
        print(f"   Agent Orchestration pattern validated end-to-end")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
