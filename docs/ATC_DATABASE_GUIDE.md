# 🗄️ ATC Database Management Guide

## Overview

This project uses a **local JSON database** (`data/atc_database.json`) that contains ATC (Anatomical Therapeutic Chemical) classifications for only the drugs used in your medications files. The database is automatically built using a **hybrid WHO + LLM approach** for maximum accuracy.

## Why This Approach?

✅ **Lightweight** - Only drugs you actually use (~350 KB for 234 drugs)  
✅ **Fast** - Instant local lookups (no web requests)  
✅ **Version-controllable** - Track in Git with your code  
✅ **Self-maintaining** - Automatically fetches new drugs as needed  
✅ **Ethical** - Respects WHO website robots.txt (10-second delay)  
✅ **Offline capable** - Works without internet after initial build  

## Initial Setup

### Step 1: Build Initial Database

Run the two-agent pipeline to extract and classify drugs:

```bash
python scripts/build_atc_database.py --medications data/medications_synthetic.csv
```

**What it does:**
1. **Agent 1 (drug_identifier)**: Extracts unique drugs from medications file
   - Creates: `data/drug_names_extracted.csv`
   - Contains: List of clean drug names (one per row)
   
2. **Agent 2 (drug_classifier)**: Classifies drugs using hybrid WHO + LLM approach:
   - Reads: `data/drug_names_extracted.csv`
   - Checks local database first (instant)
   - Fetches from WHO ATC/DDD Index (https://atcddd.fhi.no/)
   - If not found, asks LLM for synonyms and retries WHO
   - If still not found, uses LLM fallback classification (flagged for review)
   - Creates: `data/drug_classifications.csv` (detailed results with all fields)
   - Updates: `data/atc_database.json` (auto-saved after each drug classification)
   
3. Respects 10-second delay between WHO requests (per robots.txt)

**File creation order:**
1. `data/drug_names_extracted.csv` ← Agent 1
2. `data/drug_classifications.csv` ← Agent 2
3. `data/atc_database.json` ← Agent 2 (incrementally updated)

**Time estimate:** ~20 minutes for 123 drugs (10 seconds per WHO lookup)

**Final output:** `data/atc_database.json` (~2-3 KB per drug)

### Step 2: Commit to Version Control

```bash
git add data/atc_database.json
git commit -m "Add initial ATC database for medications"
```

## Runtime Behavior

### Hybrid WHO + LLM Lookup Strategy

When the pipeline encounters a drug, it uses this intelligent lookup order:

```
1. Local JSON database (instant) → Found? Use it!
   ↓ Not found
2. WHO ATC/DDD Index website (10-second delay) → Found? Save to local DB!
   ↓ Not found
3. Ask LLM for synonyms (e.g., "penicillin v" → "phenoxymethylpenicillin")
   → Retry WHO with each synonym → Found? Save with icd10_mapping_source="pending_llm"
   ↓ Still not found
4. LLM fallback classification → Flag with needs_verification=True
```

**Why this approach?**
- ✅ WHO is authoritative but limited to exact drug names
- ✅ LLM helps find drugs under different names/synonyms
- ✅ Drugs found via LLM synonyms are still WHO-sourced (authoritative)
- ✅ True LLM classifications are flagged for human review

### Automatic Database Growth

When you process a **new medications file** with unknown drugs:

1. ✅ System detects missing drug
2. ✅ Fetches ATC code from WHO website (10-second delay)
3. ✅ Saves to `data/atc_database.json` automatically
4. ✅ Next time: instant lookup!

**No manual intervention needed!**

## Database Structure

**File:** `data/atc_database.json`

**Format:** Simple dictionary with normalized drug names as keys

### Understanding `icd10_mapping_source` Field

The `icd10_mapping_source` field indicates where the ICD-10 indication data came from:

- **`"pending_llm"`**: Drug found in WHO but ICD-10 data not yet enriched (placeholder)
- **`"llm_enriched"`**: ICD-10 data filled in by LLM (reliable therapeutic indications)
- **`"llm_fallback"`**: Entire classification by LLM when not found in WHO (requires human verification)
- **`"none"`**: Unknown drug with no classification data

**Example entries:**

```json
{
  "amlodipine": {
    "code": "C08CA01",
    "drug_name": "amlodipine",
    "who_name": "amlodipine",
    "class": "Calcium channel blockers, dihydropyridine derivatives",
    "classcode": "C08CA",
    "therapeutic_category": "Calcium channel blockers",
    "anatomical_group": "C",
    "indication": "Hypertension/Angina pectoris/Coronary artery disease",
    "icd10_codes": ["I10", "I20.9", "I25.10"],
    "icd10_descriptions": {
      "I10": "Essential (primary) hypertension",
      "I20.9": "Angina pectoris, unspecified",
      "I25.10": "Atherosclerotic heart disease without angina"
    },
    "icd10_mapping_source": "llm_enriched",
    "source": "WHO ATC/DDD Index",
    "fetched_date": "2025-11-24T13:52:36.742164"
  },
  "penicillin v": {
    "code": "J01CE02",
    "drug_name": "phenoxymethylpenicillin",
    "who_name": "phenoxymethylpenicillin",
    "class": "ATC class J01CE",
    "classcode": "J01CE",
    "therapeutic_category": "Antibacterials for systemic use",
    "anatomical_group": "J",
    "indication": "To be determined by LLM",
    "icd10_codes": [],
    "icd10_descriptions": {},
    "icd10_mapping_source": "pending_llm",
    "source": "WHO ATC/DDD Index",
    "fetched_date": "2025-11-24T13:53:20.772059",
    "original_name": "penicillin v"
  },
  "unknown_drug_example": {
    "code": "A10BA02",
    "drug_name": "unknown_drug_example",
    "class": "Biguanides",
    "classcode": "A10BA",
    "therapeutic_category": "Blood glucose lowering drugs",
    "anatomical_group": "A",
    "indication": "Type 2 diabetes mellitus",
    "icd10_codes": ["E11.9"],
    "icd10_descriptions": {
      "E11.9": "Type 2 diabetes mellitus without complications"
    },
    "icd10_mapping_source": "llm_fallback",
    "needs_verification": true,
    "source": "LLM_FALLBACK",
    "fetched_date": "2025-11-24T14:00:00.000000"
  }
}
```

### Field Descriptions

| Field | Description |
|-------|-------------|
| `code` | Full ATC code (e.g., C08CA01) |
| `drug_name` | Normalized drug name used in WHO or from LLM |
| `who_name` | Name found in WHO database (always present) |
| `class` | ATC drug class description |
| `classcode` | Parent ATC class code (first 5 characters) |
| `therapeutic_category` | Therapeutic category description |
| `anatomical_group` | Single letter anatomical group code (e.g., "C", "J") |
| `indication` | Primary clinical indication |
| `icd10_codes` | List of relevant ICD-10 diagnosis codes |
| `icd10_descriptions` | Dictionary of ICD-10 codes to descriptions |
| `icd10_mapping_source` | ICD-10 data source: `pending_llm`, `llm_enriched`, or `llm_fallback` |
| `original_name` | Original input name (only present if LLM synonym was used, e.g., "penicillin v") |
| `needs_verification` | Boolean flag - only present for `llm_fallback` entries |
| `source` | Data source: "WHO ATC/DDD Index" or "LLM_FALLBACK" |
| `fetched_date` | ISO timestamp when entry was created |

## File Size Estimates

| # Drugs | Approximate Size |
|---------|-----------------|
| 100     | ~150 KB         |
| 234     | ~350 KB         |
| 500     | ~750 KB         |
| 1000    | ~1.5 MB         |

## Working with New Medications Files

### Scenario 1: Same Drugs
If your new medications file contains the same drugs:
- ✅ All lookups are instant (local database)
- ⚡ No web requests needed
- 🚀 Fast pipeline execution

### Scenario 2: Some New Drugs
If your new medications file has 10 new drugs:
- ✅ Known drugs: instant lookup
- ⏱️ New drugs: 10-second fetch per drug (100 seconds total)
- 💾 New drugs automatically saved to database
- 🔄 Next run: all drugs are instant!

### Scenario 3: Completely New Dataset
If you test with a completely different dataset:
- ⏱️ First run: fetches all new drugs (10 seconds each)
- 💾 All saved to database
- 🔄 Subsequent runs: instant lookups

## Manual Database Management

### View Database Stats

```bash
python -c "import json; db = json.load(open('data/atc_database.json')); print(f'Total drugs: {len(db)}'); print(f'Classified: {sum(1 for d in db.values() if d[\"code\"] != \"UNKNOWN\")}')"
```

### Add Single Drug Manually

```python
import json

# Load database
with open('data/atc_database.json', 'r') as f:
    db = json.load(f)

# Add new drug
db['aspirin'] = {
    "code": "N02BA01",
    "drug_name": "aspirin",
    "class": "Anilides",
    "therapeutic_category": "Analgesics",
    "anatomical_group": "N",
    "source": "Manual entry",
    "fetched_date": "2025-11-23T15:00:00"
}

# Save back
with open('data/atc_database.json', 'w') as f:
    json.dump(db, f, indent=2)
```

### Rebuild Database (Fresh Start)

```bash
# Backup current database
cp output/atc_database.json output/atc_database.backup.json

# Delete current database
rm output/atc_database.json

# Rebuild from medications file
python scripts/build_atc_database.py --medications data/medications_synthetic.csv
```

## Data Source

**Primary:** Local JSON database (`output/atc_database.json`)  
**Secondary:** WHO ATC/DDD Index (https://atcddd.fhi.no/atc_ddd_index/)  
**Fallback:** Hardcoded cache for common drugs  

### robots.txt Compliance

The WHO ATC website's `robots.txt` specifies:
```
User-agent: *
Crawl-Delay: 10
```

✅ Our implementation respects this by:
- Using 10-second delay between requests (`time.sleep(10)`)
- Minimizing requests via local database caching
- Only fetching drugs not already in database

## Troubleshooting

### Database file not found
```bash
# Create empty database directory
mkdir -p data

# Run build script
python scripts/build_atc_database.py --medications data/medications_synthetic.csv
```

### Drug not found in WHO database
- System automatically asks LLM for synonyms (e.g., "penicillin v" → "phenoxymethylpenicillin")
- Retries WHO with each synonym
- If found: marked as `icd10_mapping_source="pending_llm"` (reliable but needs ICD-10 enrichment)
- If still not found: uses LLM fallback classification with `needs_verification=true`
- Review drugs with `needs_verification=true` in the database

### Web request fails
- System falls back to local database
- If not in local database, uses hardcoded fallback
- Drug classification continues (may be "Unknown")

### Database corrupted
```bash
# Restore from backup
cp data/atc_database.backup.json data/atc_database.json

# Or rebuild from scratch
rm data/atc_database.json
python scripts/build_atc_database.py --medications data/medications_synthetic.csv
```

## Best Practices

1. ✅ **Commit `atc_database.json` to Git** - Share with team
2. ✅ **Run initial build before first pipeline run** - Avoid delays during testing
3. ✅ **Let the system auto-fetch new drugs** - Hybrid WHO + LLM approach is robust
4. ✅ **Review `pending_llm` entries** - ATC code is correct but ICD-10 mapping is incomplete
5. ✅ **Verify `llm_fallback` entries** - These require human review (check `needs_verification=true`)
6. ✅ **Review new drugs in pull requests** - Especially LLM-classified drugs
7. ✅ **Backup database before major changes** - Easy recovery

## Future Enhancements

Potential improvements for later:

- 📅 **Periodic refresh script** - Re-fetch all drugs annually (ATC codes can change)
- 📊 **Database analytics** - Show classification coverage statistics
- 🔍 **Drug name fuzzy matching** - Handle typos and variations
- 🌐 **Multiple language support** - International drug names
- 📦 **Pre-built databases** - Share common drug databases

---


