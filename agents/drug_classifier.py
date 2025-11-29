"""
Drug Classifier Agent (Self-Sufficient - Optimized Pattern)

Clean implementation with minimal agent calls:
- Python handles: Cache check, WHO lookup, database save, ATC hierarchy fetching
- Agent ONLY used for: Synonym generation, ICD-10 enrichment (LLM tasks)
- Reduces agent calls by ~70% compared to full agent orchestration

No dependencies on other agent files - completely self-contained.
"""

from google.adk import Agent
from google.adk.runners import InMemoryRunner
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
from urllib.parse import quote
from config import ATC_DATABASE_PATH

logging.basicConfig(level=logging.DEBUG)
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


def check_drug_in_database(drug_name: str, db_path: str = ATC_DATABASE_PATH) -> Dict:
    """
    Check if drug exists in database and analyze its completeness.
    
    Args:
        drug_name: Name of drug to check
        db_path: Path to ATC database
        
    Returns:
        Dict with:
        - exists: bool - whether drug exists in database
        - has_unknowns: bool - whether entry contains 'unknown' fields
        - unknown_fields: list - names of fields with 'unknown' values
        - entry: dict - the database entry (if exists)
    """
    atc_db = load_atc_database(db_path)
    entry = atc_db.get(drug_name.lower())
    
    if not entry:
        return {
            'exists': False,
            'has_unknowns': False,
            'unknown_fields': [],
            'entry': None
        }
    
    # Check for unknown/empty fields that indicate incomplete data
    unknown_fields = []
    fields_to_check = [
        'drug_class', 'therapeutic_category', 'indication',
        'icd10_codes', 'icd10_descriptions'
    ]
    
    for field in fields_to_check:
        value = entry.get(field, '')
        # Check for various forms of "unknown"
        if isinstance(value, str):
            if value.lower() in ['unknown', '', 'not found']:
                unknown_fields.append(field)
        elif isinstance(value, (list, dict)):
            if not value:  # Empty list or dict
                unknown_fields.append(field)
    
    # Also check if code is UNKNOWN or ERROR
    if entry.get('code', '') in ['UNKNOWN', 'ERROR']:
        unknown_fields.append('code')
    
    return {
        'exists': True,
        'has_unknowns': len(unknown_fields) > 0,
        'unknown_fields': unknown_fields,
        'entry': entry
    }


# ============================================================================
# WHO ATC LOOKUP FUNCTIONS
# ============================================================================

def fetch_atc_from_who(drug_name: str) -> Optional[Dict]:
    """
    Fetch ATC Information from WHO with Hierarchy Fetching.
    
    ATC Hierarchy:
    - Level 1: Anatomical main group (1 char) - e.g., C
    - Level 2: Therapeutic subgroup (3 chars) - e.g., C08 → "Calcium channel blockers"
    - Level 3: Pharmacological subgroup (4 chars) - e.g., C08C
    - Level 4: Chemical subgroup (5 chars) - e.g., C08CA → "Calcium channel blockers, dihydropyridine derivatives"
    - Level 5: Chemical substance (7 chars) - e.g., C08CA01 → "amlodipine"
    
    Example: C08CA01 → 
        Level 4 (C08CA): "Calcium channel blockers, dihydropyridine derivatives"
        Level 2 (C08): "Calcium channel blockers"
    
    """
    def try_search(search_name: str) -> Optional[BeautifulSoup]:
        """Try searching WHO with a specific name variant."""
        try:
            search_url = f"https://atcddd.fhi.no/atc_ddd_index/?name={quote(search_name)}&showdescription=no"
            logger.info(f"WHO search: {search_url}")
            resp = requests.get(search_url, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            table = soup.find("table")
            if table:
                first_row = table.find("tr")
                if first_row:
                    cells = first_row.find_all("td")
                    # Valid result: 2+ cells AND first cell looks like ATC code
                    if len(cells) >= 2:
                        atc_code = cells[0].get_text(strip=True)
                        # Check if first cell matches ATC pattern (letter + digits + letters + digits)
                        if re.match(r'^[A-Z]\d{2}[A-Z]{2}\d{2}$', atc_code):
                            logger.info(f"✓ Valid ATC result found for: {search_name}")
                            return soup
            logger.info(f"✗ No valid ATC results for: {search_name}")
            return None
        except Exception as e:
            logger.info(f"✗ Search failed for '{search_name}': {e}")
            return None
    
    try:
        # Step 1: Normalize drug name - replace "/" with space and lowercase
        normalized_name = drug_name.replace('/', ' ').lower()
        
        # Step 2: Try normalized drug name
        soup = try_search(normalized_name)
        
        # Step 3: If failed and name has 2 words, try with "and" variations
        if not soup:
            parts = normalized_name.split()
            if len(parts) == 2:
                # Try: "A B" → "A and B"
                with_and = f"{parts[0]} and {parts[1]}"
                logger.info(f"Trying with 'and': {with_and}")
                soup = try_search(with_and)
                
                # Try: "A B" → "B and A"  
                if not soup:
                    reversed_with_and = f"{parts[1]} and {parts[0]}"
                    logger.info(f"Trying reversed with 'and': {reversed_with_and}")
                    soup = try_search(reversed_with_and)
        
        if not soup:
            return None

        # Step 3: Extract ATC code from the table
        table = soup.find("table")
        if not table:
            return None
        first_row = table.find("tr")
        if not first_row:
            logger.debug(f"No first row in table for: {drug_name}")
            return None

        cells = first_row.find_all("td")
        if len(cells) < 2:
            logger.debug(f"Insufficient cells in first row for: {drug_name}")
            return None

        atc_code = cells[0].get_text(strip=True)
        who_drug_name = cells[1].get_text(strip=True)
        
        # Validate ATC code format: Letter + 2 digits + 2 letters + 2 digits (e.g., C08CA01)
        atc_pattern = r'^[A-Z]\d{2}[A-Z]{2}\d{2}$'
        if not re.match(atc_pattern, atc_code):
            logger.debug(f"Invalid ATC code format '{atc_code}' for: {drug_name}")
            return None

        # Step 3: Follow the ATC code link
        code_url = f"https://atcddd.fhi.no/atc_ddd_index/?code={atc_code}"
        code_resp = requests.get(code_url, timeout=10)
        code_resp.raise_for_status()
        code_soup = BeautifulSoup(code_resp.text, "html.parser")

        # Step 4: Parse therapeutic and chemical subgroup from page text
        # The hierarchy appears as text lines, not in a table
        drug_class = None  # Level 4 - Chemical subgroup
        drug_classcode = atc_code[:5]  # Level 4 code (5 chars)
        therapeutic_category = None  # Level 2 - Therapeutic subgroup

        # Parse hierarchy from text lines
        text = code_soup.get_text()
        for line in text.split('\n'):
            line = line.strip()
            # Level 2: 3-char code like "C08 CALCIUM CHANNEL BLOCKERS"
            if len(line) >= 4 and re.match(r'^[A-Z]\d{2}\s', line):
                if line[:3] == atc_code[:3]:
                    therapeutic_category = line[4:].strip()
            # Level 4: 5-char code like "C08CA Dihydropyridine derivatives"
            elif len(line) >= 6 and re.match(r'^[A-Z]\d{2}[A-Z]{2}\s', line):
                if line[:5] == atc_code[:5]:
                    drug_class = line[6:].strip()


        # Step 5: Build result dictionary
        result = {
            "code": atc_code,
            "who_name": who_drug_name,  # WHO's official drug name
            "drug_class": drug_class or "Unknown",  # Level 4 description
            "drug_classcode": drug_classcode or atc_code[:5],  # Level 4 code
            "therapeutic_category": therapeutic_category or "Unknown",  # Level 2 description
            "anatomical_group": atc_code[0] if atc_code else "?",
            "source": "WHO ATC/DDD Index",
            "fetched_date": pd.Timestamp.now().isoformat()
        }
        logger.debug(f"WHO lookup successful for '{drug_name}': {atc_code}")
        return result
        
    except Exception as e:
        logger.debug(f"WHO lookup failed for '{drug_name}': {e}")
        return None


# ============================================================================
# CODE EXTRACTION HELPERS
# ============================================================================

def extract_drug_classcode(atc_code: str) -> str:
    """Extract ATC Level 4 code (first 5 characters). Example: C08CA01 → C08CA"""
    return atc_code[:5] if len(atc_code) >= 5 else ""


def extract_anatomical_group(atc_code: str) -> str:
    """Extract ATC Level 1 code (first character). Example: C08CA01 → C"""
    return atc_code[0] if len(atc_code) >= 1 else ""


# ============================================================================
# AGENT CREATION
# ============================================================================

def create_drug_classifier_agent(model: str = "gemini-2.5-flash") -> Agent:
    """
    Create agent for ICD-10 enrichment and synonym generation ONLY.
    
    This agent does NOT do WHO lookups - that's handled by Python.
    Agent only provides LLM-powered tasks.
    """
    
    agent = Agent(
        model=model,
        tools=[],  # No tools needed - pure LLM knowledge
        instruction="""You are a pharmaceutical and medical coding expert.

You will be asked to perform ONE of the following tasks:

**TASK 1: Generate Drug Synonyms**
When asked to generate synonyms for a drug name, provide 2-3 international (INN) or common alternatives:
- Consider US vs international names (acetaminophen↔paracetamol)
- Consider chemical variations (adrenaline↔epinephrine)
- Consider brand vs generic names
Return ONLY a pipe-delimited list with NO extra text: synonym1|synonym2|synonym3

**TASK 2: ICD-10 Enrichment**
When given an ATC code and drug information, provide ICD-10 coding:

Required fields:
- indication: Primary therapeutic use(s), separated by /
- icd10_codes: 2-4 specific ICD-10 codes as array ["I10", "I20.9"]
- icd10_descriptions: Object mapping codes to descriptions
- indication_icd10_ranges: ICD-10 chapter ranges ["I10-I15", "I20-I25"]

Return as JSON object:
{
  "indication": "Hypertension/Angina pectoris",
  "icd10_codes": ["I10", "I20.9"],
  "icd10_descriptions": {
    "I10": "Essential (primary) hypertension",
    "I20.9": "Angina pectoris, unspecified"
  },
  "indication_icd10_ranges": ["I10-I15", "I20-I25"]
}

Perform ONLY the task requested in the prompt. Use your medical knowledge for accurate results.""",
        name="icd10_enrichment_agent",
        output_key="enrichment_data"
    )
    return agent


# ============================================================================
# CORE CLASSIFICATION FUNCTIONS
# ============================================================================

async def classify_drug(
    drug_name: str,
    atc_db_path: str = ATC_DATABASE_PATH,
    model: str = "gemini-2.5-flash"
) -> Dict:
    """
    Classify a single drug using optimized pattern with minimal agent calls.
    
    Flow:
    1. Python: Check cache → return if found (0 agent calls)
    2. Python: Try WHO lookup for drug code (0 agent calls)
    3. Python: Fetch Level 3 and Level 4 names from WHO (0 agent calls)
    4. IF FOUND: Agent enriches with ICD-10 (1 agent call)
    5. IF NOT FOUND: Agent generates synonyms (1 agent call) → Python retries WHO → Agent enriches (1 agent call)
    
    Args:
        drug_name: Name of drug to classify
        atc_db_path: Path to ATC cache database
        model: Gemini model name
        
    Returns:
        Dict with complete classification including ATC code, drug_class (Level 4), 
        therapeutic_category (Level 3), and ICD-10 codes
    """
    
    # Step 1: Check cache (Python - NO agent call)
    atc_db = load_atc_database(atc_db_path)
    cached = atc_db.get(drug_name.lower())
    if cached:
        logger.info(f"✅ Cache hit: {drug_name}")
        return cached
    
    # Step 2: Try WHO lookup (Python - NO agent call)
    who_result = fetch_atc_from_who(drug_name)
    
    if who_result:
        # FOUND in WHO! Only need ICD-10 enrichment
        logger.info(f"✅ WHO hit: {drug_name} → {who_result['code']}")
        enrichment = await _enrich_with_icd10(drug_name, who_result, model)
        result = _build_classification(drug_name, who_result, enrichment)
        
    else:
        # NOT FOUND - need synonyms
        logger.info(f"⚠️  WHO miss: {drug_name} - trying synonyms")
        synonyms = await _generate_synonyms(drug_name, model)
        
        # Try WHO with each synonym (Python - NO agent call per synonym)
        who_result = None
        for synonym in synonyms:
            logger.info(f"   Trying synonym: {synonym}")
            who_result = fetch_atc_from_who(synonym)
            if who_result:
                logger.info(f"   ✅ Found via synonym: {synonym} → {who_result['code']}")
                break
        
        if who_result:
            # Found via synonym - enrich
            enrichment = await _enrich_with_icd10(drug_name, who_result, model)
            result = _build_classification(drug_name, who_result, enrichment)
        else:
            # Not found anywhere
            logger.warning(f"   ❌ Not found: {drug_name}")
            result = _build_not_found(drug_name)
    
    # Step 3: Save to cache (Python - NO agent call)
    if result['code'] != 'UNKNOWN':
        atc_db[drug_name.lower()] = result
        save_atc_database(atc_db, atc_db_path)
    
    return result


from typing import List
import logging

logger = logging.getLogger(__name__)

async def _generate_synonyms(drug_name: str, model: str) -> List[str]:
    """Ask agent to generate synonyms (ONLY when WHO lookup fails)."""
    agent = create_drug_classifier_agent(model=model)
    runner = InMemoryRunner(agent=agent)

    prompt = f"Generate 2-3 international or common synonyms for the drug: {drug_name} and return them in a pipe-delimited string"
    response = await runner.run_debug(prompt, quiet=True)
    logger.debug(f"Response type: {type(response)}, value: {response}")

    def extract_text(obj) -> str:
        return getattr(getattr(obj, "actions", None), "state_delta", {}).get("enrichment_data", "")

    # Normalize response to a single event
    event = response[-1] if isinstance(response, list) and response else response
    text = extract_text(event)

    if text and "|" in text:
        synonyms = [s.strip() for s in text.split("|") if s.strip()]
        logger.info(f"Parsed synonyms: {synonyms[:3]}")
        return synonyms[:3]

    logger.warning("Could not parse synonyms")
    return []


async def _enrich_with_icd10(drug_name: str, who_result: Dict, model: str) -> Dict:
    """Ask agent for ICD-10 enrichment (ONLY LLM knowledge needed)."""
    agent = create_drug_classifier_agent(model=model)
    runner = InMemoryRunner(agent=agent)

    prompt = (
        f"Provide ICD-10 enrichment for:\n"
        f"Drug: {drug_name}\n"
        f"ATC Code: {who_result['code']}\n"
        f"WHO Name: {who_result['who_name']}\n\n"
        "Return JSON with: indication, icd10_codes, icd10_descriptions, indication_icd10_ranges"
    )

    response = await runner.run_debug(prompt, quiet=True)
    event = response[-1] if isinstance(response, list) and response else response
    enrichment_data = getattr(getattr(event, "actions", None), "state_delta", {}).get("enrichment_data", {})

    # Try parsing JSON if enrichment_data is a string
    if isinstance(enrichment_data, str) and enrichment_data:
        text = enrichment_data.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            text = text.rsplit("```", 1)[0]
        try:
            enrichment_data = json.loads(text.strip())
        except json.JSONDecodeError:
            logger.warning("Could not parse enrichment_data as JSON: %s", enrichment_data[:100])

    if isinstance(enrichment_data, dict) and enrichment_data:
        logger.info("ICD-10 enrichment successful: %s", list(enrichment_data.keys()))
        return enrichment_data

    logger.warning("ICD-10 enrichment returned no data")
    return {
        "indication": "Unknown",
        "icd10_codes": [],
        "icd10_descriptions": {},
        "indication_icd10_ranges": [],
    }


def _build_classification(drug_name: str, who_result: Dict, enrichment: Dict) -> Dict:
    """Build complete classification from WHO + enrichment data."""

    return {
        'code': who_result['code'],
        'drug_name': drug_name,  # Original query drug name
        'who_name': who_result['who_name'],  # WHO's official drug name
        'drug_class': who_result['drug_class'],  # Chemical Subgroup - Level 4 name from WHO
        'drug_classcode': who_result['drug_classcode'],  # Level 4 code
        'therapeutic_category': who_result['therapeutic_category'],  # Level 2 name from WHO
        'anatomical_group': who_result['anatomical_group'],
        'indication': enrichment.get('indication', 'Unknown'),
        'icd10_codes': enrichment.get('icd10_codes', []),
        'icd10_descriptions': enrichment.get('icd10_descriptions', {}),
        'indication_icd10_ranges': enrichment.get('indication_icd10_ranges', []),
        'icd10_mapping_source': 'llm_knowledge',
        'source': who_result['source'],
        'fetched_date': datetime.now().isoformat()
    }


def _build_not_found(drug_name: str) -> Dict:
    """Build NOT_FOUND result."""
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

def decide_atc_database_process(drug_name: str, update: str, db_status: Dict) -> tuple[bool, str]:
    """
    Decide whether to process a drug based on update mode and database status.
    
    Args:
        drug_name: Name of drug to check
        update: Update mode ('none', 'add_unknown', 'always')
        db_status: Dictionary returned from check_drug_in_database
        
    Returns:
        Tuple of (should_process: bool, skip_reason: str)
    """
    should_process = False
    skip_reason = ""
    
    if update == "none":
        # Only process if drug doesn't exist
        should_process = not db_status['exists']
        skip_reason = "already in database" if db_status['exists'] else ""
        
    elif update == "add_unknown":
        # Process if drug doesn't exist OR has unknown fields
        should_process = (not db_status['exists']) or db_status['has_unknowns']
        if db_status['exists'] and not db_status['has_unknowns']:
            skip_reason = "complete entry exists"
        elif db_status['exists'] and db_status['has_unknowns']:
            logger.info(f"    ♻️  Updating incomplete entry (unknown fields: {', '.join(db_status['unknown_fields'])})")
        
    elif update == "always":
        # Always process
        should_process = True
        if db_status['exists']:
            logger.info(f"    ♻️  Re-fetching (update mode: always)")
    
    else:
        raise ValueError(f"Invalid update mode: {update}. Must be 'none', 'add_unknown', or 'always'")
    
    return should_process, skip_reason


def process_drug_names_file(
    input_file: str,
    output_file: str,
    error_log: str = "logs/drug_classifier_errors.log",
    model: str = "gemini-2.5-flash",
    update: str = "none"
) -> pd.DataFrame:
    """
    Process drug names file and classify all drugs using optimized pattern.
    
    Only calls agent when:
    - WHO lookup fails (need synonyms)
    - Need ICD-10 enrichment
    
    Reduces costs by ~70% compared to full agent orchestration!
    
    Args:
        input_file: Path to CSV with drug_name column
        output_file: Path for output CSV
        error_log: Path for error log
        model: Gemini model to use
        update: Update mode for existing database entries:
            - "none" (default): Only add new drugs, skip existing ones
            - "add_unknown": Update entries that have 'unknown' fields
            - "always": Always update/re-fetch all drugs
        
    Returns:
        DataFrame with classification results
    """
    logger.info("="*70)
    logger.info("DRUG CLASSIFIER - OPTIMIZED (Minimal Agent Calls)")
    logger.info("="*70)
    logger.info(f"📥 Input: {input_file}")
    logger.info(f"📤 Output: {output_file}")
    logger.info(f"🔄 Update mode: {update}")
    
    df = pd.read_csv(input_file)
    logger.info(f"📊 Before starting the process {len(df)} drugs with duplicates")
    df = df.drop_duplicates(subset=['drug_name'])
    logger.info(f"📊 Processing {len(df)} drugs after removing duplicates")
    
    results = []
    errors = []
    agent_calls = 0
    cache_hits = 0
    who_hits = 0
    
    async def process_all():
        nonlocal agent_calls, cache_hits, who_hits
        
        for idx, row in df.iterrows():
            drug_name = row['drug_name']
            try:
                logger.info(f"[{idx+1}/{len(df)}] 🔬 {drug_name}")
                
                # Check database status and decide whether to process
                db_status = check_drug_in_database(drug_name, ATC_DATABASE_PATH)
                should_process, skip_reason = decide_atc_database_process(drug_name, update, db_status)
                
                # Skip if not processing
                if not should_process:
                    logger.info(f"    ⏭️  Skipping: {skip_reason}")
                    cache_hits += 1
                    
                    # Use existing entry for output
                    existing = db_status['entry']
                    output_row = {
                        'drug_name': drug_name,
                        'atc_code': existing['code'],
                        'atc_class': existing['drug_class'],
                        'indication': existing['indication'],
                        'icd10_codes': ', '.join(existing['icd10_codes']) if isinstance(existing['icd10_codes'], list) else existing['icd10_codes'],
                        'icd10_mapping_source': existing.get('icd10_mapping_source', ''),
                        'source': existing['source'],
                        'needs_verification': existing['code'] in ['UNKNOWN', 'ERROR'],
                        'comment': 'from_local_ATC_database'
                    }
                    results.append(output_row)
                    continue
                
                # MAIN ENTRY POINT - Process the drug
                result = await classify_drug(
                    drug_name=drug_name,
                    atc_db_path=ATC_DATABASE_PATH,
                    model=model
                )
                
                # Track stats
                if result.get('source') == 'NOT_FOUND':
                    agent_calls += 2  # synonyms + enrichment attempt
                elif 'cache' in str(result):
                    cache_hits += 1
                else:
                    agent_calls += 1  # ICD-10 enrichment only
                    who_hits += 1
                
                # Format output
                output_row = {
                    'drug_name': drug_name,
                    'atc_code': result['code'],
                    'atc_class': result['drug_class'],
                    'indication': result['indication'],
                    'icd10_codes': ', '.join(result['icd10_codes']) if isinstance(result['icd10_codes'], list) else result['icd10_codes'],
                    'icd10_mapping_source': result.get('icd10_mapping_source', ''),
                    'source': result['source'],
                    'needs_verification': result['code'] in ['UNKNOWN', 'ERROR'],
                    'comment': ''
                }
                results.append(output_row)
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"    ❌ Error: {error_msg}")
                
                # Check for quota exhausted error and abort
                if '429' in error_msg and 'RESOURCE_EXHAUSTED' in error_msg:
                    logger.error("\n" + "="*70)
                    logger.error("🛑 QUOTA EXHAUSTED - Aborting processing")
                    logger.error("="*70)
                    logger.error("Gemini API quota exceeded. Please check:")
                    logger.error("  - https://ai.google.dev/gemini-api/docs/rate-limits")
                    logger.error("  - https://ai.dev/usage?tab=rate-limit")
                    logger.error(f"\n⏸️  Processed {idx + 1}/{len(df)} drugs before quota exhausted")
                    raise RuntimeError(f"API quota exhausted after processing {idx + 1} drugs") from e
                
                errors.append(f"{drug_name}: {error_msg}")
                results.append({
                    'drug_name': drug_name,
                    'atc_code': 'ERROR',
                    'atc_class': '',
                    'indication': '',
                    'icd10_codes': '',
                    'icd10_mapping_source': '',
                    'source': 'ERROR',
                    'needs_verification': True,
                    'comment': error_msg
                })
    
    asyncio.run(process_all())
    
    # Save results
    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results_df.to_csv(output_file, index=False)
    
    # Save error log
    if errors:
        os.makedirs(os.path.dirname(error_log), exist_ok=True)
        with open(error_log, 'w') as f:
            f.write('\n'.join(errors))
    
    # Summary statistics
    successful = len(results_df[~results_df['atc_code'].isin(['UNKNOWN', 'ERROR'])])
    unknown = len(results_df[results_df['atc_code'] == 'UNKNOWN'])
    error_count = len(results_df[results_df['atc_code'] == 'ERROR'])
    
    logger.info("="*70)
    logger.info(f"✅ Success: {successful} | ⚠️  Unknown: {unknown} | ❌ Errors: {error_count}")
    logger.info(f"💰 Cost Optimization (update mode: {update}):")
    logger.info(f"   Cache hits/skipped: {cache_hits} (0 agent calls)")
    logger.info(f"   WHO hits: {who_hits} (1 agent call each)")
    logger.info(f"   Total agent calls: {agent_calls}")
    logger.info(f"📤 Output: {output_file}")
    logger.info("="*70)
    
    return results_df
