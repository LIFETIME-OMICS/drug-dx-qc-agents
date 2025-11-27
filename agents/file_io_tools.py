"""
File I/O Tools for Google ADK Agents

These tools allow agents to read from and write to CSV files.
Following the FunctionTool pattern from Google ADK.
"""

import pandas as pd
import os
from typing import Dict, Any
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
        df.to_csv(file_path, index=False)
        
        success_msg = f"Successfully wrote {len(df)} rows to {file_path}"
        logger.info(success_msg)
        return success_msg
        
    except Exception as e:
        error_msg = f"ERROR writing to {file_path}: {str(e)}"
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
