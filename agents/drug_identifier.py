"""
Drug Identifier Agent (Proper Google ADK Architecture with InMemoryRunner)

ADK pattern (Kaggle Day 1b):
1. Tools are pure Python functions (no LLM calls)
2. Agent is created with model + instruction + tools
3. InMemoryRunner executes the agent with await runner.run_debug()
4. Agent reasons about when to use tools and validates extractions

User Requirements:
- Identify which columns contain drug information (don't assume fixed format)
- Verify that regex extraction results are correct
- Handle flexible CSV formats
"""

from google.adk import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
import os
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging
import asyncio
from .drug_extraction_tools import extract_drug_name_regex, extract_drug_names_from_list
from .file_io_tools import read_csv_file, write_csv_file, write_dataframe_to_csv
from config import TMP_DIR, LOGS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# TOOL FUNCTIONS (Pure Python - No LLM)
# ============================================================================

def get_csv_columns(file_path: str) -> List[str]:
    """
    Get list of column names from a CSV file.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        List of column names
    """
    df = pd.read_csv(file_path, nrows=0)
    return df.columns.tolist()


def get_csv_sample(file_path: str, num_rows: int = 5) -> str:
    """
    Get sample rows from CSV file for inspection.
    
    Args:
        file_path: Path to CSV file
        num_rows: Number of sample rows to return
        
    Returns:
        String representation of sample data
    """
    df = pd.read_csv(file_path, nrows=num_rows)
    return df.to_string()


def get_column_data_sample(file_path: str, column_name: str, num_samples: int = 10) -> List[str]:
    """
    Get sample values from a specific column.
    
    Args:
        file_path: Path to CSV file
        column_name: Name of column to sample
        num_samples: Number of samples to return
        
    Returns:
        List of sample values from the column
    """
    df = pd.read_csv(file_path)
    if column_name not in df.columns:
        return [f"ERROR: Column '{column_name}' not found"]
    
    samples = df[column_name].dropna().unique()[:num_samples]
    return samples.tolist()


def validate_drug_extraction(original: str, extracted: str) -> Dict[str, any]:
    """
    Validate that drug extraction looks reasonable.
    
    Args:
        original: Original medication description
        extracted: Extracted drug name
        
    Returns:
        Dict with validation results: {is_valid: bool, issues: List[str]}
    """
    issues = []
    
    if not extracted or extracted.strip() == "":
        issues.append("Empty extraction")
    elif len(extracted) < 3:
        issues.append("Very short drug name (< 3 chars)")
    elif extracted.lower() == "unknown":
        issues.append("Failed to identify drug")
    elif any(f" {word} " in f" {extracted.lower()} " or f" {word}," in f" {extracted.lower()}" 
             for word in ["tablet", "capsule", "injection", "oral", "mg", "ml"]):
        # Check for whole words only (with spaces or commas around them)
        issues.append("Contains dosage/formulation terms")
    elif extracted.lower() == original.lower():
        # Only flag if extraction is exactly the same (case-insensitive)
        issues.append("Extraction unchanged from original")
    
    return {
        "is_valid": len(issues) == 0,
        "issues": issues
    }


# ============================================================================
# AGENT CREATION
# ============================================================================

def create_drug_identifier_agent(model: str = "gemini-2.5-flash") -> Agent:
    """
    Create a Drug Identifier Agent following proper Google ADK architecture.
    
    Architecture:
    - TOOLS: Pure Python functions (extract_drug_name_regex)
    - AGENT: LLM (Gemini) + Tools
    - The agent REASONS about when/how to use tools
    
    Args:
        model: Gemini model to use
        
    Returns:
        Configured Google ADK Agent
    """
    
    # Define tools (pure functions, no LLM)
    # FunctionTool wraps functions - docstrings provide descriptions to the agent
    
    columns_tool = FunctionTool(get_csv_columns)
    sample_tool = FunctionTool(get_csv_sample)
    column_sample_tool = FunctionTool(get_column_data_sample)
    extraction_tool = FunctionTool(extract_drug_name_regex)
    validation_tool = FunctionTool(validate_drug_extraction)
    
    # File I/O tools for reading input and writing output
    read_csv_tool = FunctionTool(read_csv_file)
    write_csv_tool = FunctionTool(write_csv_file)
    write_df_tool = FunctionTool(write_dataframe_to_csv)
    
    # Create agent with tools
    agent = Agent(
        model=model,
        name="drug_identifier",
        instruction="""You are a pharmaceutical expert specializing in drug name extraction from CSV files.

Your role has two phases:

PHASE 1 - COLUMN DETECTION (when given a CSV file path):
- Use get_csv_columns() to see all available columns
- Use get_csv_sample() to inspect sample data
- Use get_column_data_sample() to examine specific columns
- Identify which column contains medication/drug descriptions
- Report your findings clearly

PHASE 2 - DRUG EXTRACTION (when given medication descriptions):
- Use extract_drug_name_regex() to extract clean drug names
- Use validate_drug_extraction() to verify each extraction
- IMPORTANT: Use your pharmaceutical knowledge to verify and correct the regex output
- The regex may make mistakes - YOU must fix them!

Key Guidelines:
- Extract active pharmaceutical ingredient names only
- Remove dosages, formulations, routes, brand names
- For combination drugs, keep all ingredients (e.g., "acetaminophen oxycodone")
- For salt forms of drugs, remove the salt ONLY if it's not the drug itself:
  * "naproxen sodium" → "naproxen" (sodium is the salt form)
  * "sodium chloride" → "sodium chloride" (sodium IS the drug, keep it!)
  * "potassium chloride" → "potassium chloride" (potassium IS the drug, keep it!)
- For product/brand names, identify the actual active ingredient:
  * "Camila 28 Day Pack" → regex returns "camila", but YOU know Camila is norethindrone
  * Use your knowledge to return the correct active ingredient
- NEVER put explanatory text in drug_name - only the actual drug name
- If validation fails, explain the issue clearly

File I/O Operations:
- Use read_csv_file() to read input CSV files
- Use write_csv_file() to write output CSV (provide CSV-formatted string)
- Use write_dataframe_to_csv() to write output from dictionary format

Examples:
- "amLODIPine 2.5 MG Oral Tablet" → "amlodipine"
- "Acetaminophen 325 MG / Oxycodone Hydrochloride 5 MG" → "acetaminophen oxycodone"
- "Naproxen sodium 220 MG" → "naproxen" (remove salt)
- "Sodium Chloride 0.9% Injectable" → "sodium chloride" (sodium IS the drug, keep it!)
- "Camila 28 Day Pack" → "norethindrone" (use your knowledge to identify the active ingredient)
""",
        tools=[columns_tool, sample_tool, column_sample_tool, extraction_tool, validation_tool, 
               read_csv_tool, write_csv_tool, write_df_tool],
        output_key="drug_names_extracted"
    )
    
    return agent


# ============================================================================
# AGENT EXECUTION WITH INMEMORYRUNNER
# ============================================================================

async def detect_medication_column(
    input_file: str,
    model: str = "gemini-2.5-flash"
) -> Tuple[str, str]:
    """
    Use agent to detect which column contains medication descriptions.
    
    Args:
        input_file: Path to CSV file
        model: Gemini model to use
        
    Returns:
        Tuple of (column_name, agent_reasoning)
    """
    agent = create_drug_identifier_agent(model=model)
    runner = InMemoryRunner(agent=agent)
    
    prompt = f"""Analyze this CSV file and identify which column contains medication/drug descriptions:

File: {input_file}

Steps:
1. Use get_csv_columns() to see all columns
2. Use get_csv_sample() to see sample data
3. Use get_column_data_sample() to inspect specific columns
4. Determine which column contains medication descriptions

Provide your answer in this format:
COLUMN: [column_name]
REASONING: [your reasoning]
"""
    
    print(f"\n🔍 PHASE 1: Detecting medication column in {input_file}...")
    response = await runner.run_debug(prompt)
    
    # Extract column name from response
    response_text = response.text if hasattr(response, 'text') else str(response)
    
    # Parse response to find column name
    column_name = None
    for line in response_text.split('\n'):
        if line.startswith('COLUMN:'):
            column_name = line.replace('COLUMN:', '').strip()
            break
    
    if not column_name:
        # Fallback: try common column names
        df = pd.read_csv(input_file, nrows=0)
        for col in ['DESCRIPTION', 'description', 'medication', 'drug', 'drug_name']:
            if col in df.columns:
                column_name = col
                response_text = f"Fallback detection used: Found column '{col}'"
                break
    
    if not column_name:
        raise ValueError(f"Could not detect medication column in {input_file}")
    
    return column_name, response_text


async def extract_and_validate_drug(
    description: str,
    model: str = "gemini-2.5-flash"
) -> Dict[str, any]:
    """
    Use agent to extract and validate a single drug name.
    
    NOTE: This is an optional validation feature controlled by use_agent_validation parameter
    in process_medications_file(). Production typically uses regex extraction (PHASE 2) only,
    without agent validation (PHASE 3). Set use_agent_validation=True to enable.
    
    Args:
        description: Medication description text
        model: Gemini model to use
        
    Returns:
        Dict with: {drug_name: str, is_valid: bool, issues: List[str], reasoning: str}
    """
    agent = create_drug_identifier_agent(model=model)
    runner = InMemoryRunner(agent=agent)
    
    prompt = f"""Extract the drug name from this medication description and validate it:

Description: "{description}"

Steps:
1. Use extract_drug_name_regex() to extract the drug name
2. Use validate_drug_extraction() to verify it looks correct
3. Report the extracted name and validation results

Provide your answer in this format:
DRUG_NAME: [extracted drug name]
VALID: [yes/no]
ISSUES: [list any issues, or "none"]
"""
    
    response = await runner.run_debug(prompt)
    response_text = response.text if hasattr(response, 'text') else str(response)
    
    # Parse response
    drug_name = ""
    is_valid = True
    issues = []
    
    for line in response_text.split('\n'):
        if line.startswith('DRUG_NAME:'):
            drug_name = line.replace('DRUG_NAME:', '').strip()
        elif line.startswith('VALID:'):
            is_valid = 'yes' in line.lower()
        elif line.startswith('ISSUES:'):
            issues_str = line.replace('ISSUES:', '').strip()
            if issues_str.lower() not in ['none', '']:
                issues.append(issues_str)
    
    return {
        'drug_name': drug_name,
        'is_valid': is_valid,
        'issues': issues,
        'reasoning': response_text
    }


async def process_medications_file(
    input_file: str,
    output_file: str = "tmp/drug_names_extracted.csv",
    error_log: str = None,
    model: str = "gemini-2.5-flash",
    use_agent_validation: bool = False
) -> pd.DataFrame:
    """
    ASYNC PROCESSING: Extract drug names from medications file using InMemoryRunner.
    
    This is the main entry point for the drug_identifier agent.
    Uses proper Google ADK InMemoryRunner pattern.
    
    Workflow:
    1. PHASE 1: Agent detects which column contains medications
    2. PHASE 2: Extract drug names from each unique description
    3. PHASE 3 (optional): Agent validates extractions
    4. Save results to CSV with columns: [drug_description, drug_name, comment]
    
    Args:
        input_file: Path to medications CSV (e.g., data/medications_synthetic.csv)
        output_file: Path to output CSV with extracted drug names
        error_log: Path to error log file (defaults to logs/drug_identifier_errors.log)
        model: Gemini model to use
        use_agent_validation: Whether to use agent for validation (slower but more accurate)
        
    Returns:
        DataFrame with columns: [drug_description, drug_name, comment]
        
    Output Files:
        - {output_file}: CSV with extracted drug names
        - {error_log}: Log file with extraction errors
    """
    # Set defaults
    if error_log is None:
        error_log = os.path.join(LOGS_DIR, "drug_identifier_errors.log")
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(error_log), exist_ok=True)
    
    # Setup error logging
    error_handler = logging.FileHandler(error_log)
    error_handler.setLevel(logging.ERROR)
    logger.addHandler(error_handler)
    
    logger.info(f"Starting drug extraction from {input_file}")
    print(f"\n{'='*70}")
    print(f"🔬 DRUG IDENTIFIER AGENT - InMemoryRunner Pattern")
    print(f"{'='*70}")
    print(f"📥 Input:  {input_file}")
    print(f"📤 Output: {output_file}")
    print(f"📝 Errors: {error_log}")
    print(f"🤖 Model:  {model}")
    
    # PHASE 1: Detect medication column
    try:
        medication_col, detection_reasoning = await detect_medication_column(input_file, model)
        print(f"\n✅ DETECTED MEDICATION COLUMN: '{medication_col}'")
        print(f"📝 Agent reasoning:\n{detection_reasoning}\n")
    except Exception as e:
        logger.error(f"Failed to detect medication column: {e}")
        raise
    
    # Read medications file
    df = pd.read_csv(input_file)
    
    if medication_col not in df.columns:
        raise ValueError(f"Detected column '{medication_col}' not found in {input_file}")
    
    # Get unique descriptions
    unique_descriptions = df[medication_col].unique()
    print(f"\n📊 Found {len(df)} total medication records")
    print(f"📊 Found {len(unique_descriptions)} unique drug descriptions")
    print(f"🎯 Efficiency: Processing {len(unique_descriptions)} instead of {len(df)}\n")
    
    # PHASE 2 & 3: Extract and validate drug names
    print(f"🔬 PHASE 2: Extracting drug names...")
    results = []
    error_count = 0
    
    for i, description in enumerate(unique_descriptions, 1):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(unique_descriptions)} ({i/len(unique_descriptions)*100:.1f}%)")
        
        try:
            if use_agent_validation:
                # Use agent for extraction and validation (slower)
                result = await extract_and_validate_drug(description, model)
                drug_name = result['drug_name']
                
                if not result['is_valid']:
                    comment = "VALIDATION FAILED: " + "; ".join(result['issues'])
                    error_count += 1
                    logger.warning(f"Validation failed for '{description}': {comment}")
                else:
                    comment = ""
            else:
                # Fast path: direct extraction with validation
                drug_name = extract_drug_name_regex(description)
                validation = validate_drug_extraction(description, drug_name)
                
                if not validation['is_valid']:
                    comment = "; ".join(validation['issues'])
                    error_count += 1
                    logger.warning(f"Validation issues for '{description}': {comment}")
                else:
                    comment = ""
            
            results.append({
                'drug_description': description,
                'drug_name': drug_name if drug_name else "UNKNOWN",
                'comment': comment
            })
            
        except Exception as e:
            error_count += 1
            error_msg = f"EXCEPTION: {str(e)}"
            logger.error(f"Exception processing '{description}': {e}")
            print(f"❌ EXCEPTION: {description} - {e}")
            
            results.append({
                'drug_description': description,
                'drug_name': "ERROR",
                'comment': error_msg
            })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    results_df.to_csv(output_file, index=False)
    
    # Summary
    successful = len(results_df[results_df['comment'] == ''])
    issues = len(results_df[results_df['comment'] != ''])
    
    print(f"\n{'='*70}")
    print(f"✅ DRUG EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"📊 Total processed: {len(results_df)}")
    print(f"✅ Successful: {successful}")
    print(f"⚠️  With issues: {issues}")
    print(f"\n📤 Results saved to: {output_file}")
    print(f"📝 Error log saved to: {error_log}")
    
    if issues > 0:
        print(f"\n⚠️  {issues} drugs had validation issues - review {error_log}")
    
    logger.info(f"Extraction complete: {successful} successful, {issues} with issues")
    
    return results_df
