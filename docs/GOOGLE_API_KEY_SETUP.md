# 🔑 Google API Key Setup Guide

## Overview

To run the Google ADK agents (drug-identifier, drug-classifier, etc.), you need a **Gemini API key** from Google AI Studio.

---

## Step 1: Get Your API Key

1. **Go to Google AI Studio**
   - Visit: https://aistudio.google.com/apikey
   - Sign in with your Google account

2. **Create API Key**
   - Click "Create API key"
   - Select "Create API key in new project" or choose an existing project
   - Copy the API key (starts with `AIza...`)

3. **⚠️ Important**: 
   - Keep this key secret! Don't commit it to git
   - The free tier includes generous limits for testing

---

## Step 2: Set Up Environment Variable

### Windows (PowerShell)

```powershell
# Temporary (current session only)
$env:GOOGLE_API_KEY = "your-api-key-here"

# Permanent (add to your PowerShell profile)
[System.Environment]::SetEnvironmentVariable('GOOGLE_API_KEY', 'your-api-key-here', 'User')
```

### Windows (Command Prompt)

```cmd
setx GOOGLE_API_KEY "your-api-key-here"
```

### Linux/Mac

```bash
# Add to ~/.bashrc or ~/.zshrc
export GOOGLE_API_KEY="your-api-key-here"

# Then reload
source ~/.bashrc
```

---

## Step 3: Create .env File (Recommended) ✅

**This is the recommended approach for local development!**

1. Copy the example file:
   ```powershell
   Copy-Item .env.example .env
   ```

2. Edit `.env` and add your actual API key:
   ```bash
   GOOGLE_API_KEY=
   ```

3. The scripts automatically load from `.env`:
   ```python
   from dotenv import load_dotenv
   load_dotenv()  # Already included in all scripts!
   ```

**✅ Benefits**:
- No need to set environment variable each session
- `.env` is in `.gitignore` (safe from accidental commits)
- `.env.example` is committed (shows required variables)

**⚠️ Important**: Never commit `.env` to git! Only commit `.env.example`

---

## Step 4: Verify Setup

Run the built-in verification script:

```powershell
python scripts/check_api_key.py
```

This will:
- ✅ Check if `GOOGLE_API_KEY` is set
- ✅ Verify the API key works with Gemini
- ✅ Test agent connection
- ✅ Display helpful error messages if something is wrong

---

## API Key Best Practices

### ✅ DO:
- Store in environment variables or `.env` file
- Add `.env` to `.gitignore`
- Use separate keys for development and production
- Monitor your usage at https://aistudio.google.com/

### ❌ DON'T:
- Hardcode API keys in your source code
- Commit API keys to version control
- Share your API keys publicly
- Use production keys for testing

---

## Model Recommendations

| Model | Status | Free Tier | Speed | Accuracy |
|-------|--------|-----------|-------|----------|
| `gemini-2.5-flash` | ✅ **Recommended** | Generous | Fast | High |
| `gemini-2.5-pro` | ✅ Works | Good | Slower | Highest |
| `gemini-2.0-flash-exp` | ⚠️ Limited quota | Very limited | Fast | High |

**Default**: All agents use `gemini-2.5-flash` or DEFAULT_MODEL from config.py

## Free Tier Limits (as of 2025)

- **Rate Limits**: 15 requests per minute (RPM)
- **Quota**: 1,500 requests per day
- **Models**: Access to Gemini 1.5 Flash, 1.5 Pro, 2.5 Flash

For our drug extraction (~123 drugs), this is well within free tier limits.

---

## Troubleshooting

### Error: "GOOGLE_API_KEY not found"

```powershell
# Check if set
$env:GOOGLE_API_KEY

# If empty, set it
$env:GOOGLE_API_KEY = "your-key-here"
```

### Error: "Invalid API key"

- Verify your key at https://aistudio.google.com/apikey
- Make sure there are no extra spaces or quotes
- Try regenerating the key

### Error: "Rate limit exceeded"

- You're making too many requests too quickly
- Add delays between requests (we already do this in the code)
- Upgrade to paid tier if needed

---

## Cost Estimation

For our project (123 drugs):

- **Input tokens**: ~50 tokens per drug × 123 = ~6,150 tokens
- **Output tokens**: ~10 tokens per drug × 123 = ~1,230 tokens
- **Total**: ~7,380 tokens

At free tier prices: **$0.00** (well within limits)

---

## Next Steps

After setting up your API key, test the pipeline:

1. **Test with 3 drugs** (fast test):
   ```powershell
   python scripts/build_atc_database.py --medications data/medications_test.csv
   ```

2. **Build full ATC database** (20 minutes for 123 drugs):
   ```powershell
   python scripts/build_atc_database.py --medications data/medications_synthetic.csv
   ```

The pipeline uses a **hybrid approach**:
- ✅ WHO ATC database lookups (authoritative, but slow - 10 sec delay per drug)
- ✅ LLM synonym suggestions (helps find drugs not in WHO)
- ✅ LLM fallback classification (when WHO lookup fails)
- ✅ Auto-saved to `data/atc_database.json` for instant future lookups
