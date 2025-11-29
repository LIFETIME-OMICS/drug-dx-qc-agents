"""
Drug Name Extraction Tools

Pure Python functions that can be used as tools by Google ADK agents.
These functions do NOT call LLMs - they are deterministic tools.
"""

import re
from typing import List


def extract_drug_name_regex(description: str) -> str:
    """
    Extract drug name from medication description using regex patterns.
    
    This is a TOOL function - it does NOT call LLMs.
    It's a deterministic function that the LLM agent can use.
    
    Args:
        description: Raw medication description
        
    Returns:
        Clean drug name(s) in lowercase
    """
    if not description or not isinstance(description, str):
        return ""
    
    normalized = description.lower()
    
    # Remove NDA codes (e.g., "nda020800")
    normalized = re.sub(r'nda\d+\s*', '', normalized)
    
    # Remove dosage amounts with units (e.g., "10 MG", "2.5 ML", "200 actuat")
    normalized = re.sub(r'\d+\.?\d*\s*(mg|ml|mcg|g|actuat|unt|hr)\b', '', normalized, flags=re.IGNORECASE)
    
    # Remove standalone numbers
    normalized = re.sub(r'\b\d+\s*', '', normalized)
    
    # Remove unit abbreviations that might remain (ml, mg, etc.)
    normalized = re.sub(r'\b(mg|ml|mcg|g|actuat|unt|hr)\b', '', normalized, flags=re.IGNORECASE)
    
    # Remove formulation types
    formulations = [
        'oral tablet', 'oral capsule', 'tablet', 'capsule',
        'injection', 'injectable', 'prefilled syringe', 'auto-injector',
        'metered dose inhaler', 'dry powder inhaler', 'inhaler', 'inhalant',
        'transdermal patch', 'transdermal system', 'transdermal',
        'extended release', 'abuse-deterrent',
        'chewable', 'sublingual', 'buccal',
        'intrauterine system', 'vaginal ring', 'drug implant',
        'mucosal spray', 'topical', 'cream', 'ointment',
        'solution', 'suspension', 'drops'
    ]
    
    for formulation in formulations:
        normalized = normalized.replace(formulation, ' ')
    
    # Remove pack information (e.g., "28 day pack", "21 day")
    normalized = re.sub(r'\b\d+\s*day\s*pack\b', '', normalized)
    
    # Remove route information
    normalized = re.sub(r'\b(oral|injectable|topical|inhalation|nasal|rectal|vaginal|ophthalmic)\b', '', normalized)
    
    # Handle combination drugs - keep both active ingredients
    # Convert "/" to space (e.g., "acetaminophen / oxycodone" → "acetaminophen oxycodone")
    normalized = normalized.replace('/', ' ')
    
    # Remove brand names in brackets
    normalized = re.sub(r'\[.*?\]', '', normalized)
    
    # Remove extra whitespace and clean up
    normalized = ' '.join(normalized.split()).strip()
    
    # Remove common salt forms and chemical suffixes
    salt_forms = [
        'hydrochloride', 'hydrocholoride', 'sulfate', 'sodium', 'potassium',
        'phosphate', 'bitartrate', 'succinate', 'maleate', 'fumarate',
        'tartrate', 'citrate', 'acetate', 'chloride', 'bromide',
        'hydrobromide', 'calcium', 'magnesium'
    ]
    
    words = normalized.split()
    words = [w for w in words if w not in salt_forms]
    
    # Remove common non-drug words that might remain
    stop_words = ['the', 'a', 'an', 'and', 'or', 'of', 'for', 'in', 'on', 'at']
    words = [w for w in words if w not in stop_words]
    
    normalized = ' '.join(words)
    
    return normalized.strip()


def extract_drug_names_from_list(descriptions: List[str]) -> List[str]:
    """
    Extract drug names from a list of descriptions.
    
    This is a TOOL function that processes multiple descriptions.
    
    Args:
        descriptions: List of medication descriptions
        
    Returns:
        List of clean drug names
    """
    drug_names = []
    for desc in descriptions:
        name = extract_drug_name_regex(desc)
        if name:
            drug_names.append(name)
    
    # Remove duplicates and sort
    return sorted(list(set(drug_names)))
