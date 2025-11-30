"""
File I/O Tools for Google ADK Agents

These tools allow agents to read from and write to CSV files.
Following the FunctionTool pattern from Google ADK.
"""

import pandas as pd
import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def read_csv_file(file_path: str) -> str:
    """
    Read a CSV file and return its contents as a formatted string.
    
    Args:
        file_path: Path to the CSV file to read
        
    Returns:
        String representation of the CSV contents with metadata
        
    Example:
        content = read_csv_file("data/medications.csv")
    """
    try:
        if not os.path.exists(file_path):
            return f"ERROR: File not found: {file_path}"
        
        df = pd.read_csv(file_path)
        
        # Return formatted information about the file
        result = f"Successfully read {file_path}\n"
        result += f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n"
        result += f"Columns: {', '.join(df.columns.tolist())}\n\n"
        result += "First 5 rows:\n"
        result += df.head().to_string()
        
        logger.info(f"Read CSV file: {file_path} ({df.shape[0]} rows)")
        return result
        
    except Exception as e:
        error_msg = f"ERROR reading {file_path}: {str(e)}"
        logger.error(error_msg)
        return error_msg


def write_csv_file(file_path: str, data: str) -> str:
    """
    Write data to a CSV file. Data should be in CSV format (comma-separated).
    
    Args:
        file_path: Path where the CSV file should be written
        data: CSV-formatted string data to write
        
    Returns:
        Confirmation message or error message
        
    Example:
        result = write_csv_file(
            "output/results.csv",
            "patient_id,drug_name,dose\\n1,aspirin,100mg\\n2,ibuprofen,200mg"
        )
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Parse the CSV string and write as DataFrame
        from io import StringIO
        df = pd.read_csv(StringIO(data))
        
        df.to_csv(file_path, index=False)
        
        success_msg = f"Successfully wrote {len(df)} rows to {file_path}"
        logger.info(success_msg)
        return success_msg
        
    except Exception as e:
        error_msg = f"ERROR writing to {file_path}: {str(e)}"
        logger.error(error_msg)
        return error_msg


def write_dataframe_to_csv(file_path: str, dataframe_dict: Dict[str, Any]) -> str:
    """
    Write a dictionary representation of a DataFrame to a CSV file.
    
    This is a helper for agents that construct data programmatically.
    For QC evaluator output, automatically enforces the correct column order.
    
    Args:
        file_path: Path where the CSV file should be written
        dataframe_dict: Dictionary with column names as keys and lists as values
        
    Returns:
        Confirmation message or error message
        
    Example:
        result = write_dataframe_to_csv(
            "output/results.csv",
            {
                "patient_id": [1, 2, 3],
                "drug_name": ["aspirin", "ibuprofen", "metformin"],
                "dose": ["100mg", "200mg", "500mg"]
            }
        )
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        df = pd.DataFrame(dataframe_dict)
        
        # If this looks like QC evaluator output, enforce correct column order
        expected_qc_columns = [
            'patient_id', 'encounter_id', 'drug_name', 'drug_description', 
            'atc_code', 'drug_class', 'expected_icd10_codes', 'expected_icd10_ranges', 
            'actual_icd10_codes', 'status', 'match_type', 'matched_codes', 'reason'
        ]
        
        # Check if this DataFrame has the QC evaluator columns
        if all(col in df.columns for col in expected_qc_columns):
            # Reorder columns to match expected order
            df = df[expected_qc_columns]
            logger.info(f"Reordered QC evaluator columns to standard format")
        
        df.to_csv(file_path, index=False)
        
        success_msg = f"Successfully wrote {len(df)} rows to {file_path}"
        logger.info(success_msg)
        return success_msg
        
    except Exception as e:
        error_msg = f"ERROR writing to {file_path}: {str(e)}"
        logger.error(error_msg)
        return error_msg


def append_row_to_csv(file_path: str, row_dict: Dict[str, Any]) -> str:
    """
    Append a single row to a CSV file, creating the file with headers if it doesn't exist.
    
    This enables incremental writing - useful for long-running processes where you want
    to save results progressively instead of accumulating everything in memory.
    
    Args:
        file_path: Path to the CSV file
        row_dict: Dictionary with column names as keys and values for this row
        
    Returns:
        Confirmation message or error message
        
    Example:
        result = append_row_to_csv(
            "output/results.csv",
            {
                "patient_id": "P001",
                "drug_name": "aspirin",
                "status": "PASS",
                "reason": "Appropriate for hypertension"
            }
        )
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Create DataFrame from single row
        df = pd.DataFrame([row_dict])
        
        # If this looks like QC evaluator output, enforce correct column order
        expected_qc_columns = [
            'patient_id', 'encounter_id', 'drug_name', 'drug_description', 
            'atc_code', 'drug_class', 'expected_icd10_codes', 'expected_icd10_ranges', 
            'actual_icd10_codes', 'status', 'match_type', 'matched_codes', 'reason'
        ]
        
        # Enforce column order if this is QC evaluator output
        if all(col in df.columns for col in expected_qc_columns):
            df = df[expected_qc_columns]
        
        # Check if file exists
        file_exists = os.path.exists(file_path)
        
        if file_exists:
            # Append mode - don't write header
            df.to_csv(file_path, mode='a', header=False, index=False)
            success_msg = f"Successfully appended 1 row to {file_path}"
        else:
            # New file - write with header
            df.to_csv(file_path, mode='w', header=True, index=False)
            success_msg = f"Successfully created {file_path} and wrote 1 row"
        
        logger.info(success_msg)
        return success_msg
        
    except Exception as e:
        error_msg = f"ERROR appending to {file_path}: {str(e)}"
        logger.error(error_msg)
        return error_msg


def get_csv_info(file_path: str) -> Dict[str, Any]:
    """
    Get metadata about a CSV file without loading all data.
    
    This is useful for agents to understand file structure before processing.
    
    Args:
        file_path: Path to the CSV file to inspect
        
    Returns:
        String with file metadata (row count, column count, column names)
        
    Example:
        info = get_csv_info("data/medications.csv")
        # Output: "File: data/medications.csv\nTotal rows: 58\nTotal columns: 13\nColumns: patient, encounter, description, ..."
    """
    try:
        if not os.path.exists(file_path):
            return f"ERROR: File not found: {file_path}"
        
        df = pd.read_csv(file_path)
        
        result = f"File: {file_path}\n"
        result += f"Total rows: {len(df)}\n"
        result += f"Total columns: {len(df.columns)}\n"
        result += f"Columns: {', '.join(df.columns.tolist())}"
        
        logger.info(f"Got CSV info: {file_path} ({len(df)} rows, {len(df.columns)} columns)")
        return result
        
    except Exception as e:
        error_msg = f"ERROR getting info for {file_path}: {str(e)}"
        logger.error(error_msg)
        return error_msg


def read_csv_batch(file_path: str, start_row: int, batch_size: int = 10) -> str:
    """
    Read a specific batch of rows from a CSV file.
    
    This allows agents to process large CSV files in manageable chunks.
    
    Args:
        file_path: Path to the CSV file to read
        start_row: Starting row index (0-based)
        batch_size: Number of rows to read (default: 10)
        
    Returns:
        String representation of the batch with metadata
        
    Example:
        batch = read_csv_batch("data/medications.csv", start_row=0, batch_size=10)
        # Returns rows 0-9
        
        batch = read_csv_batch("data/medications.csv", start_row=10, batch_size=10)
        # Returns rows 10-19
    """
    try:
        if not os.path.exists(file_path):
            return f"ERROR: File not found: {file_path}"
        
        df = pd.read_csv(file_path)
        total_rows = len(df)
        
        # Handle edge cases
        if start_row >= total_rows:
            return f"ERROR: start_row ({start_row}) is beyond file length ({total_rows} rows)"
        
        # Calculate end row
        end_row = min(start_row + batch_size, total_rows)
        
        # Get batch
        batch_df = df.iloc[start_row:end_row]
        
        # Format result
        result = f"Successfully read batch from {file_path}\n"
        result += f"Batch: rows {start_row} to {end_row-1} (of {total_rows} total rows)\n"
        result += f"Rows in this batch: {len(batch_df)}\n"
        result += f"Columns: {', '.join(batch_df.columns.tolist())}\n\n"
        result += "Batch data:\n"
        result += batch_df.to_string()
        
        logger.info(f"Read CSV batch: {file_path} rows {start_row}-{end_row-1}")
        return result
        
    except Exception as e:
        error_msg = f"ERROR reading batch from {file_path}: {str(e)}"
        logger.error(error_msg)
        return error_msg


def append_to_csv(file_path: str, data: str) -> str:
    """
    Append data to an existing CSV file.
    
    Args:
        file_path: Path to the CSV file to append to
        data: CSV-formatted string data to append (should match existing columns)
        
    Returns:
        Confirmation message or error message
    """
    try:
        from io import StringIO
        new_df = pd.read_csv(StringIO(data))
        
        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df
        
        combined_df.to_csv(file_path, index=False)
        
        success_msg = f"Successfully appended {len(new_df)} rows to {file_path} (total: {len(combined_df)} rows)"
        logger.info(success_msg)
        return success_msg
        
    except Exception as e:
        error_msg = f"ERROR appending to {file_path}: {str(e)}"
        logger.error(error_msg)
        return error_msg


def reorder_csv_rows(
    input_file: str,
    reference_file: str,
    patient_col_input: str = 'patient_id',
    encounter_col_input: str = 'encounter_id',
    description_col_input: str = 'drug_description',
    patient_col_ref: str = 'patient',
    encounter_col_ref: str = 'encounter',
    description_col_ref: str = 'description'
) -> pd.DataFrame:
    """
    Reorder input CSV to match the row order of a reference medications file.
    
    This ensures QC evaluation output matches the exact order of medications
    in the input file, even if the agent processed them out of order.
    
    Matches on patient + encounter + description to handle multiple medications
    per encounter correctly.
    
    Reorders only the rows that exist in the input file - no validation of missing rows.
    
    Args:
        input_file: Path to file to be reordered (e.g., qc_flags.csv)
        reference_file: Path to reference medications CSV (defines order)
        patient_col_input: Patient ID column name in input file (default: 'patient_id')
        encounter_col_input: Encounter ID column name in input file (default: 'encounter_id')
        description_col_input: Drug description column name in input file (default: 'drug_description')
        patient_col_ref: Patient column name in reference file (default: 'patient')
        encounter_col_ref: Encounter column name in reference file (default: 'encounter')
        description_col_ref: Description column name in reference file (default: 'description')
        
    Returns:
        Reordered DataFrame matching reference file order (only rows that exist in input)
        
    Example:
        # Reorder QC output to match medications input order
        reordered_df = reorder_csv_rows(
            input_file="output/qc_flags.csv",
            reference_file="input/medications.csv"
        )
    """
    try:
        # Load files
        input_df = pd.read_csv(input_file)
        reference_df = pd.read_csv(reference_file)
        
        logger.info(f"Reordering {len(input_df)} rows to match reference file order")
        
        # Create merge key for reference file (patient + encounter + description)
        # This handles multiple medications per encounter
        reference_df['_merge_key'] = (
            reference_df[patient_col_ref].astype(str) + '|' + 
            reference_df[encounter_col_ref].astype(str) + '|' +
            reference_df[description_col_ref].astype(str)
        )
        reference_df['_original_order'] = range(len(reference_df))
        
        # Create merge key for input file
        input_df['_merge_key'] = (
            input_df[patient_col_input].astype(str) + '|' + 
            input_df[encounter_col_input].astype(str) + '|' +
            input_df[description_col_input].astype(str)
        )
        
        # Merge to get original order for each input row
        merged_df = input_df.merge(
            reference_df[['_merge_key', '_original_order']],
            on='_merge_key',
            how='left'
        )
        
        # Sort by original order
        reordered_df = merged_df.sort_values('_original_order')
        
        # Drop merge columns
        reordered_df = reordered_df.drop(columns=['_merge_key', '_original_order'])
        
        # Save reordered file (overwrites input file with reordered version)
        reordered_df.to_csv(input_file, index=False)
        
        success_msg = f"Successfully reordered {len(reordered_df)} rows in {input_file}"
        logger.info(success_msg)
        
        return reordered_df
        
    except Exception as e:
        error_msg = f"ERROR reordering {input_file}: {str(e)}"
        logger.error(error_msg)
        raise
