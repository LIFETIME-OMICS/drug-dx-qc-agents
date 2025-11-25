"""
Drug Classifier Agent (Pure Function Pattern - Google ADK)

Classifies drugs to ATC codes using WHO database + LLM enrichment.

ADK Pattern (following drug_identifier.py):
1. Tools are pure Python functions
2. Agent created with model + instruction + tools  
3. InMemoryRunner executes agent with await runner.run_debug()
4. No classes, state passed as function parameters
"""

from google.adk import Agent
from google.adk.tools import FunctionTool
import os
import json
import pandas as pd
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import logging
import asyncio
import re
from datetime import datetime
from config import ATC_DATABASE_PATH, DATA_DIR, TMP_DIR, LOGS_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def load_atc_database(db_path: str = ATC_DATABASE_PATH) -> Dict:
    """Load ATC database from JSON file."""
    if os.path.exists(db_path):
        try:
            with open(db_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading ATC database: {e}")
            return {}
    return {}


def save_atc_database(atc_database: Dict, db_path: str = ATC_DATABASE_PATH) -> None:
    """Save ATC database to JSON file."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with open(db_path, 'w', encoding='utf-8') as f:
        json.dump(atc_database, f, indent=2, ensure_ascii=False)


# ============================================================================
# WHO ATC LOOKUP FUNCTIONS
# ============================================================================

def fetch_atc_from_who(drug_name: str) -> Optional[Dict]:
    """Fetch ATC code from WHO ATC/DDD Index."""
    import re
    try:
        url = f"https://atcddd.fhi.no/atc_ddd_index/?name={drug_name}&showdescription=no"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        rows = soup.find_all('tr')
        for row in rows:
            cells = row.find_all('td')
            if len(cells) >= 2:
                code_cell = cells[0].get_text(strip=True)
                name_cell = cells[1].get_text(strip=True)
                
                # Validate ATC code format: Letter + 2 digits + 2 letters + 2 digits (e.g., C08CA01)
                # This rejects garbage like "Implant" which has 7 chars but wrong format
                atc_pattern = r'^[A-Z]\d{2}[A-Z]{2}\d{2}$'
                if re.match(atc_pattern, code_cell):
                    return {'code': code_cell, 'name': name_cell, 'source': 'WHO'}
        
        logger.debug(f"Not found in WHO ATC database: {drug_name}")
        return None
    except Exception as e:
        logger.debug(f"WHO lookup failed for '{drug_name}': {e}")
        return None


def extract_drug_class(atc_code: str) -> str:
    """Extract drug class from ATC code."""
    if len(atc_code) != 7:
        return "Unknown"
    groups = {'A': 'Alimentary tract and metabolism', 'B': 'Blood and blood forming organs', 'C': 'Cardiovascular system', 'D': 'Dermatologicals', 'G': 'Genito-urinary system and sex hormones', 'H': 'Systemic hormonal preparations', 'J': 'Antiinfectives for systemic use', 'L': 'Antineoplastic and immunomodulating agents', 'M': 'Musculo-skeletal system', 'N': 'Nervous system', 'P': 'Antiparasitic products', 'R': 'Respiratory system', 'S': 'Sensory organs', 'V': 'Various'}
    return groups.get(atc_code[0], 'Unknown')


def extract_therapeutic_category(atc_code: str) -> str:
    """Extract therapeutic category from ATC code."""
    if len(atc_code) >= 3:
        return f"ATC {atc_code[:3]}"
    return "Unknown"


# ============================================================================
# AGENT CREATION
# ============================================================================

def create_drug_classifier_agent(model: str = "gemini-2.5-flash") -> Agent:
    """Create drug classifier agent with tools."""
    agent = Agent(
        model=model,
        tools=[FunctionTool(fetch_atc_from_who), FunctionTool(extract_drug_class), FunctionTool(extract_therapeutic_category)],
        instruction="""You are a pharmaceutical classification expert specializing in ATC classification.

When given a drug name:
1. Use fetch_atc_from_who() to search WHO database
2. If found, extract class information
3. Provide 2-3 relevant ICD-10 codes for the drug's primary indications

Return JSON: {"code": "C08CA01", "class": "Drug class", "therapeutic_category": "Category", "indication": "Primary indication", "icd10_codes": ["I10", "I20.9"], "icd10_descriptions": {"I10": "Essential hypertension", "I20.9": "Angina pectoris"}}""",
        name="drug_classifier"
    )
    return agent


# ============================================================================
# ASYNC PROCESSING FUNCTIONS
# ============================================================================

async def suggest_synonyms_async(drug_name: str, model: str = "gemini-2.5-flash") -> List[str]:
    """
    Ask LLM for drug name synonyms.
    
    Uses direct LLM call (not agent) to avoid tool restrictions.
    """
    import google.genai as genai
    
    prompt = f"""What are the international (INN) or common synonyms for the drug "{drug_name}"?

Consider:
- US vs international names (e.g., acetaminophen vs paracetamol)
- Brand names vs generic names  
- Chemical variations (e.g., adrenaline vs epinephrine)

Return ONLY a JSON array of 2-3 most likely synonyms:
["synonym1", "synonym2", "synonym3"]

If no synonyms exist, return empty array: []
"""
    
    try:
        client = genai.Client()
        response = client.models.generate_content(model=model, contents=prompt)
        response_text = response.text.strip()
        
        # Parse JSON response
        json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
        if json_match:
            synonyms = json.loads(json_match.group(0))
            logger.info(f"LLM suggested synonyms for '{drug_name}': {synonyms}")
            return synonyms
        
        logger.warning(f"No JSON array in synonym response for '{drug_name}'")
        return []
        
    except Exception as e:
        logger.error(f"Error getting synonyms for '{drug_name}': {e}")
        return []


async def enrich_icd10_async(drug_name: str, atc_code: str, drug_class: str, model: str = "gemini-2.5-flash") -> Dict:
    """
    Enrich drug data with ICD-10 codes and indications.
    
    Uses direct LLM call (not agent) to avoid tool restrictions.
    """
    from google import genai
    
    client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
    
    prompt = f"""For drug "{drug_name}" (ATC: {atc_code}, Class: {drug_class}):

Provide:
1. indication: Primary therapeutic indication(s) separated by /
2. icd10_codes: 2-4 specific ICD-10 codes as a list
3. icd10_descriptions: Dictionary mapping each ICD-10 code to its description
4. indication_icd10_ranges: ICD-10 chapter/ranges (e.g., ["I10-I15", "I20-I25"])

Return ONLY valid JSON (no markdown, no explanation):
{{
  "indication": "...",
  "icd10_codes": ["I10", "I20.9"],
  "icd10_descriptions": {{"I10": "Essential (primary) hypertension", "I20.9": "Angina pectoris, unspecified"}},
  "indication_icd10_ranges": ["I10-I15", "I20-I25"]
}}"""
    
    response = client.models.generate_content(model=model, contents=prompt)
    response_text = response.text
    logger.debug(f"ICD-10 enrichment raw response: {response_text[:200]}")
    response_text = response_text.strip().replace('```json', '').replace('```', '').strip()
    
    try:
        return json.loads(response_text)
    except Exception as e:
        logger.warning(f"Failed to parse ICD-10 enrichment: {e}")
        return {'indication': 'Unknown', 'icd10_codes': [], 'icd10_descriptions': {}, 'indication_icd10_ranges': []}


async def classify_single_drug_async(drug_name: str, atc_database: Dict = None, atc_db_path: str = ATC_DATABASE_PATH, model: str = "gemini-2.5-flash") -> Dict:
    """
    Classify a single drug using WHO database + LLM enrichment.
    
    Strategy:
    1. Check local database cache
    2. Try WHO ATC database lookup
    3. If not found, try synonym lookup + WHO
    4. Enrich result with ICD-10 codes via LLM
    """
    if atc_database is None:
        atc_database = load_atc_database(atc_db_path)
    
    drug_key = drug_name.lower().strip()
    
    # Check database cache
    if drug_key in atc_database:
        logger.info(f" Found in database: '{drug_name}'")
        return atc_database[drug_key]
    
    # Try WHO lookup
    logger.info(f" WHO lookup: '{drug_name}'")
    who_data = fetch_atc_from_who(drug_name)
    
    if who_data:
        atc_code = who_data['code']
        who_name = who_data['name']
        drug_class = extract_drug_class(atc_code)
        therapeutic_category = extract_therapeutic_category(atc_code)
        enriched = await enrich_icd10_async(drug_name, atc_code, drug_class, model)
        
        # Match production database structure exactly
        result = {
            'code': atc_code,
            'drug_name': drug_name,
            'who_name': who_name,
            'drug_class': drug_class,
            'drug_classcode': atc_code[:5] if len(atc_code) >= 5 else atc_code,
            'therapeutic_category': therapeutic_category,
            'anatomical_group': atc_code[0] if atc_code else '',
            'indication': enriched.get('indication', ''),
            'icd10_codes': enriched.get('icd10_codes', []),
            'icd10_descriptions': enriched.get('icd10_descriptions', {}),
            'indication_icd10_ranges': enriched.get('indication_icd10_ranges', []),
            'icd10_mapping_source': 'llm_enriched',
            'source': 'WHO ATC/DDD Index',
            'fetched_date': datetime.now().isoformat()
        }
        atc_database[drug_key] = result
        save_atc_database(atc_database, atc_db_path)
        return result
    
    # Try synonyms
    logger.info(f" Trying synonyms: '{drug_name}'")
    synonyms = await suggest_synonyms_async(drug_name, model)
    for synonym in synonyms:
        who_data = fetch_atc_from_who(synonym)
        if who_data:
            logger.info(f" Found via synonym '{synonym}'")
            atc_code = who_data['code']
            who_name = who_data['name']
            drug_class = extract_drug_class(atc_code)
            therapeutic_category = extract_therapeutic_category(atc_code)
            enriched = await enrich_icd10_async(drug_name, atc_code, drug_class, model)
            
            # Match production database structure exactly
            result = {
                'code': atc_code,
                'drug_name': drug_name,
                'who_name': who_name,
                'drug_class': drug_class,
                'drug_classcode': atc_code[:5] if len(atc_code) >= 5 else atc_code,
                'therapeutic_category': therapeutic_category,
                'anatomical_group': atc_code[0] if atc_code else '',
                'indication': enriched.get('indication', ''),
                'icd10_codes': enriched.get('icd10_codes', []),
                'icd10_descriptions': enriched.get('icd10_descriptions', {}),
                'indication_icd10_ranges': enriched.get('indication_icd10_ranges', []),
                'icd10_mapping_source': 'llm_enriched',
                'source': f'WHO ATC/DDD Index (via synonym: {synonym})',
                'fetched_date': datetime.now().isoformat()
            }
            atc_database[drug_key] = result
            save_atc_database(atc_database, atc_db_path)
            return result
    
    # Not found - match production structure
    from datetime import datetime
    logger.warning(f"  Not found: '{drug_name}'")
    return {
        'code': 'UNKNOWN',
        'drug_name': drug_name,
        'who_name': '',
        'drug_class': 'Unknown',
        'drug_classcode': '',
        'therapeutic_category': 'Unknown',
        'anatomical_group': '',
        'indication': 'Unknown',
        'icd10_codes': [],
        'icd10_descriptions': {},
        'indication_icd10_ranges': [],
        'icd10_mapping_source': '',
        'source': 'NOT_FOUND',
        'fetched_date': datetime.now().isoformat()
    }


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def process_drug_names_file(input_file: str = "data/drug_names_extracted.csv", output_file: str = "data/drug_classifications.csv", error_log: str = "logs/drug_classifier_errors.log", model: str = "gemini-2.5-flash") -> pd.DataFrame:
    """Process drug names file and classify all drugs."""
    logger.info("="*70)
    logger.info("DRUG CLASSIFIER - BATCH PROCESSING")
    logger.info("="*70)
    logger.info(f" Input: {input_file}")
    logger.info(f" Output: {output_file}")
    
    df = pd.read_csv(input_file)
    logger.info(f" Processing {len(df)} drugs")
    atc_database = load_atc_database()
    agent = create_drug_classifier_agent(model=model)
    results = []
    errors = []
    
    async def process_all():
        for idx, row in df.iterrows():
            drug_name = row['drug_name']
            try:
                logger.info(f"[{idx+1}/{len(df)}] Processing: {drug_name}")
                result = await classify_single_drug_async(drug_name, atc_database, ATC_DATABASE_PATH, agent, model)
                output_row = {
                    'drug_name': drug_name,
                    'atc_code': result['code'],
                    'atc_class': result['drug_class'],
                    'indication': result['indication'],
                    'icd10_codes': ', '.join(result['icd10_codes']),
                    'icd10_mapping_source': result['icd10_mapping_source'],
                    'source': result['source'],
                    'needs_verification': result['code'] in ['UNKNOWN', 'ERROR'],
                    'comment': ''
                }
                results.append(output_row)
            except Exception as e:
                logger.error(f"Error: {e}")
                errors.append(f"{drug_name}: {str(e)}")
                results.append({'drug_name': drug_name, 'atc_code': 'ERROR', 'atc_class': '', 'indication': '', 'icd10_codes': '', 'icd10_mapping_source': '', 'source': 'ERROR', 'needs_verification': True, 'comment': str(e)})
    
    asyncio.run(process_all())
    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results_df.to_csv(output_file, index=False)
    
    if errors:
        os.makedirs(os.path.dirname(error_log), exist_ok=True)
        with open(error_log, 'w') as f:
            f.write('\n'.join(errors))
    
    successful = len(results_df[~results_df['atc_code'].isin(['UNKNOWN', 'ERROR'])])
    logger.info("="*70)
    logger.info(f" Success: {successful} |   Unknown: {len(results_df[results_df['atc_code'] == 'UNKNOWN'])} |  Errors: {len(results_df[results_df['atc_code'] == 'ERROR'])}")
    logger.info(f" Output: {output_file}")
    
    return results_df