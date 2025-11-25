"""
Stats Summarizer Agent

This agent summarizes medication statistics across patients (not per patient)
by drug classes and diagnoses/conditions to provide aggregate insights.
It can answer to user's prompts and needs session control.
"""

from typing import Dict, Tuple
import pandas as pd


class StatsSummarizer:
    """
    Agent responsible for summarizing medication statistics across patients
    by drug classes and diagnoses/conditions.
    
    Produces aggregate tables showing:
    - Drug class usage across all patients
    - Diagnosis-drug concordance rates
    - Diagnosis-drug discordance rates
    """
    
    def __init__(self):
        """Initialize the Stats Summarizer agent."""
        self.summary_stats = {}
    
    def summarize_drug_class_usage(self, classified_df: pd.DataFrame) -> pd.DataFrame:
        """
        Summarize drug class usage across all patients.
        
        Creates a table with:
        - drug_class: ATC drug class name
        - patient_count: Number of unique patients using drugs in this class
        - total_prescriptions: Total number of prescriptions in this class
        
        Args:
            classified_df: DataFrame with classified drugs (must have 'drug_class' and 'patient' columns)
            
        Returns:
            DataFrame with drug class usage statistics
        """
        if 'drug_class' not in classified_df.columns or 'patient' not in classified_df.columns:
            raise ValueError("DataFrame must contain 'drug_class' and 'patient' columns")
        
        summary = classified_df.groupby('drug_class').agg(
            patient_count=('patient', 'nunique'),
            total_prescriptions=('patient', 'count')
        ).reset_index()
        
        summary = summary.sort_values('patient_count', ascending=False)
        
        return summary
    
    def summarize_diagnosis_drug_concordance(
        self, 
        medication_df: pd.DataFrame,
        diagnosis_df: pd.DataFrame,
        concordance_flags_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Summarize diagnosis-drug concordance across all patients.
        
        Creates a table with:
        - drug_class: Drug class or drug name
        - expected_diagnosis: Expected diagnosis for the drug
        - concordant_patients: Number of patients with matching diagnosis
        - total_patients_on_drug: Total patients prescribed this drug
        - concordance_rate: Percentage of patients with proper diagnosis
        
        Args:
            medication_df: DataFrame with medication records
            diagnosis_df: DataFrame with diagnosis records
            concordance_flags_df: DataFrame from QC evaluation with concordance flags
            
        Returns:
            DataFrame with concordance statistics
        """
        # TODO: Implement concordance summarization logic
        # Group by drug_class and expected_diagnosis
        # Count patients where has_matching_diagnosis == True
        pass
    
    def summarize_diagnosis_drug_discordance(
        self,
        medication_df: pd.DataFrame,
        diagnosis_df: pd.DataFrame,
        concordance_flags_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Summarize diagnosis-drug discordance across all patients.
        
        Creates a table with:
        - drug_class: Drug class or drug name
        - expected_diagnosis: Expected diagnosis for the drug
        - discordant_patients: Number of patients WITHOUT matching diagnosis
        - total_patients_on_drug: Total patients prescribed this drug
        - discordance_rate: Percentage of patients lacking proper diagnosis
        
        Args:
            medication_df: DataFrame with medication records
            diagnosis_df: DataFrame with diagnosis records
            concordance_flags_df: DataFrame from QC evaluation with concordance flags
            
        Returns:
            DataFrame with discordance statistics
        """
        # TODO: Implement discordance summarization logic
        # Group by drug_class and expected_diagnosis
        # Count patients where has_matching_diagnosis == False
        pass
    
    def generate_comprehensive_summary(
        self,
        medication_df: pd.DataFrame,
        diagnosis_df: pd.DataFrame,
        classified_df: pd.DataFrame,
        concordance_flags_df: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate all summary statistics.
        
        Args:
            medication_df: DataFrame with medication records
            diagnosis_df: DataFrame with diagnosis records
            classified_df: DataFrame with classified drugs
            concordance_flags_df: DataFrame from QC evaluation
            
        Returns:
            Dictionary containing:
            - 'drug_class_usage': Drug class usage table
            - 'concordance': Diagnosis-drug concordance table
            - 'discordance': Diagnosis-drug discordance table
        """
        summaries = {
            'drug_class_usage': self.summarize_drug_class_usage(classified_df),
            'concordance': self.summarize_diagnosis_drug_concordance(
                medication_df, diagnosis_df, concordance_flags_df
            ),
            'discordance': self.summarize_diagnosis_drug_discordance(
                medication_df, diagnosis_df, concordance_flags_df
            )
        }
        
        self.summary_stats = summaries
        return summaries
    
    def print_summary(self, summaries: Dict[str, pd.DataFrame]) -> None:
        """
        Print formatted summary statistics.
        
        Args:
            summaries: Dictionary of summary DataFrames
        """
        print("\n" + "=" * 80)
        print("DRUG CLASS USAGE SUMMARY")
        print("=" * 80)
        print(summaries['drug_class_usage'].to_string(index=False))
        
        print("\n" + "=" * 80)
        print("DIAGNOSIS-DRUG CONCORDANCE SUMMARY")
        print("=" * 80)
        print(summaries['concordance'].to_string(index=False))
        
        print("\n" + "=" * 80)
        print("DIAGNOSIS-DRUG DISCORDANCE SUMMARY")
        print("=" * 80)
        print(summaries['discordance'].to_string(index=False))
