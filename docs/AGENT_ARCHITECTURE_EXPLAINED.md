# 🤖 Agent Architecture

## Overview

This project uses a **two-agent pipeline** to classify drugs from medications data into WHO ATC codes.

```
medications.csv → Agent 1 → drug_names.csv → Agent 2 → classifications.csv + atc_database.json
```

---

## Agent 1: Drug Identifier

**Purpose**: Extract clean drug names from messy medication descriptions

**Input**: `medications.csv` (e.g., "Amlodipine 5mg tablet once daily")

**Process**:
- Removes dosage information (5mg, 10mg, etc.)
- Removes administration instructions (once daily, twice daily)
- Removes brand names and product codes
- Extracts just the drug name (Amlodipine)

**Output**: `data/drug_names_extracted.csv`
```csv
drug_name
amlodipine
lisinopril
metformin
```

**Technology**: Regex-based extraction (fast, deterministic)

---

## Agent 2: Drug Classifier

**Purpose**: Classify drugs to WHO ATC codes using hybrid WHO + LLM approach

**Input**: `data/drug_names_extracted.csv`

**Process** (Hybrid WHO + LLM):
1. **Check local database** → Found? Use it! (instant)
2. **WHO ATC/DDD Index lookup** → Found? Save and use! (10 sec delay)
3. **LLM synonym suggestion** → Retry WHO with synonyms
4. **LLM fallback** → Classify with AI (flagged for verification)

**Outputs**:
- `data/drug_classifications.csv` - Detailed results with all fields
- `data/atc_database.json` - Persistent database (incrementally updated)

**Technology**: 
- WHO web scraping (authoritative source)
- Google Gemini 2.5-flash (synonym suggestion + fallback)

---

## Intermediate Files

| File | Created By | Purpose | Keep? |
|------|------------|---------|-------|
| `drug_names_extracted.csv` | Agent 1 | Clean drug names | No (temporary) |
| `drug_classifications.csv` | Agent 2 | Detailed classification results | No (temporary) |
| `atc_database.json` | Agent 2 | Persistent drug → ATC mapping | **Yes** (commit to git) |

**Note**: Intermediate files are automatically cleaned up on each pipeline run. Only `atc_database.json` persists.

---

## Data Flow Example

**Input medication**:
```
"Amlodipine 5mg tablet, take 1 tablet by mouth once daily"
```

**After Agent 1**:
```csv
drug_name
amlodipine
```

**After Agent 2**:
```json
{
  "amlodipine": {
    "code": "C08CA01",
    "drug_name": "amlodipine",
    "class": "Calcium channel blockers, dihydropyridine derivatives",
    "mapping_source": "direct",
    "source": "WHO ATC/DDD Index",
    "indication": "Hypertension/Angina pectoris"
  }
}
```

---

## Running the Pipeline

```bash
# Process test data (3 drugs, ~1 minute)
python scripts/build_atc_database.py --medications data/medications_test.csv

# Process full dataset (123 drugs, ~20 minutes)
python scripts/build_atc_database.py --medications data/medications_synthetic.csv
```

**What happens**:
1. 🧹 Cleans up intermediate files from previous runs
2. 🔬 Agent 1 extracts drug names
3. 🏥 Agent 2 classifies drugs with hybrid WHO + LLM
4. 💾 Updates `atc_database.json` after each drug
5. ✅ Shows summary statistics

---

## Database Growth Strategy

**First run** (empty database):
- Fetches all drugs from WHO (10 sec delay each)
- LLM helps with synonyms when WHO fails
- Saves everything to `atc_database.json`

**Subsequent runs** (with database):
- Known drugs: instant lookup (no web requests)
- New drugs: fetch and add to database
- Database grows automatically over time

---

## Mapping Source Types

| Value | Meaning | Reliability |
|-------|---------|-------------|
| `direct` | Found directly in WHO database | ✅ High (authoritative) |
| `pending_llm` | Found via LLM-suggested synonym in WHO but ICD-10 data not yet enriched (placeholder)| ✅ High (WHO-sourced) |
| `llm_fallback` | Classified by LLM (not in WHO) | ⚠️ Medium (needs verification) |
| `llm_enriched` | Classified by LLM (not in WHO) | ⚠️ Medium (needs verification) |
| `none` | Unknown drug with no classification data | NA |


**Review strategy**: 
- `direct` and `pending_llm`: Generally reliable
- `llm_fallback`: Check `needs_verification` flag, validate with domain expert

---

## For More Details

- **Database structure**: See `docs/ATC_DATABASE_GUIDE.md`
- **API key setup**: See `docs/GOOGLE_API_KEY_SETUP.md`
- **Batch processing**: See `docs/BATCH_PROCESSING_GUIDE.md`
