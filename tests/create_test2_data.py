"""
Create test2 data by extracting 10 random patients from synthetic data.
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Read source data
meds_df = pd.read_csv('data/medications_synthetic.csv')
conds_df = pd.read_csv('data/conditions_synthetic.csv')

# Get 10 random unique patients from medications
all_patients = meds_df['patient'].unique()
patients = np.random.choice(all_patients, size=10, replace=False)
print(f"Selected {len(patients)} random patients for test2 (seed=42)")

# Filter both datasets for these patients
meds_test2 = meds_df[meds_df['patient'].isin(patients)].copy()
conds_test2 = conds_df[conds_df['patient'].isin(patients)].copy()

print(f"Medications rows: {len(meds_test2)}")
print(f"Conditions rows: {len(conds_test2)}")

# Create output directory if it doesn't exist
output_dir = Path('tests/test2/input2')
output_dir.mkdir(parents=True, exist_ok=True)

# Save to CSV
meds_test2.to_csv(output_dir / 'medications_test.csv', index=False)
conds_test2.to_csv(output_dir / 'conditions_test.csv', index=False)

print(f"\nSaved files to {output_dir}")
print(f"  - medications_test.csv: {len(meds_test2)} rows")
print(f"  - conditions_test.csv: {len(conds_test2)} rows")
