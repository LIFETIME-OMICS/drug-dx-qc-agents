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
CACHE_DIR = PROJECT_ROOT / "cache"
LOGS_DIR = PROJECT_ROOT / "logs"

# ATC Database location (single source of truth)
ATC_DATABASE_PATH = str(OUTPUT_DIR / "atc_database.json")

# Default file paths (production synthetic data)
DEFAULT_MEDICATIONS_FILE = str(DATA_DIR / "medications_synthetic.csv")
DEFAULT_CONDITIONS_FILE = str(DATA_DIR / "conditions_synthetic.csv")

# Example/Demo file paths (test1: 3-patient dataset - used as fallback for CLI demos)
TEST_MEDICATIONS_FILE = "tests/test1/input1/medications_test.csv"
TEST_CONDITIONS_FILE = "tests/test1/input1/conditions_test.csv"

# Output file paths (production)
QC_FLAGS_OUTPUT = str(OUTPUT_DIR / "qc_flags.csv")
DRUG_CLASSIFICATIONS_OUTPUT = str(OUTPUT_DIR / "drug_classifications.csv")

# Ensure directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
TMP_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
