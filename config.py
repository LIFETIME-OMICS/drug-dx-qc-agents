"""
Project Configuration Constants

Centralized configuration to avoid hardcoding paths across multiple files.
Following DRY principle - define constants once, import everywhere.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
TMP_DIR = PROJECT_ROOT / "tmp"
LOGS_DIR = PROJECT_ROOT / "logs"
EXAMPLE_DIR = PROJECT_ROOT / "examples"

# ATC Database location (single source of truth)
ATC_DATABASE_PATH = str(OUTPUT_DIR / "atc_database.json")

# Default file paths (production synthetic data)
DEFAULT_MEDICATIONS_FILE = str(DATA_DIR / "medications_synthetic.csv")
DEFAULT_CONDITIONS_FILE = str(DATA_DIR / "conditions_synthetic.csv")

# Example/Demo file paths (test: 8-patient dataset - used as fallback for CLI demos)
TEST_MEDICATIONS_FILE = "data/medications_synthetic_short_8.csv"
TEST_CONDITIONS_FILE = "data/conditions_synthetic_short_8.csv"
TEST_DRUG_NAMES_FILE = str(EXAMPLE_DIR / "drug_names_extracted_short.csv")
TEST_DRUG_CLASSIFICATIONS_FILE = str(EXAMPLE_DIR / "drug_classifications_short.csv")
TEST_QC_FLAGS_FILE = str(EXAMPLE_DIR / "example_qc_flags.csv")


# Output file paths (production)
QC_FLAGS_OUTPUT = str(OUTPUT_DIR / "qc_flags.csv")
DRUG_CLASSIFICATIONS_OUTPUT = str(OUTPUT_DIR / "drug_classifications.csv")

# Model configuration
DEFAULT_MODEL = "gemini-2.0-flash"  # Use 2.0-flash version when out of tokens (use gemini-2.5-flash for latest)

# Ensure directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
TMP_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
