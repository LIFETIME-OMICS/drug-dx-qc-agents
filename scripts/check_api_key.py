"""
Check Google API Key Setup

This script verifies that your Google API key is properly configured
and can connect to Gemini models.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def check_api_key():
    """Check if GOOGLE_API_KEY is set."""
    print("=" * 70)
    print("🔑 Checking Google API Key Setup")
    print("=" * 70)
    
    # Check environment variable
    api_key = os.getenv('GOOGLE_API_KEY')
    
    if not api_key:
        print("\n❌ GOOGLE_API_KEY not found in environment variables!")
        print("\n📖 Setup Instructions:")
        print("   1. Go to: https://aistudio.google.com/apikey")
        print("   2. Create an API key")
        print("   3. Set environment variable (PowerShell):")
        print("      $env:GOOGLE_API_KEY = 'your-api-key-here'")
        print("\n📚 See docs/GOOGLE_API_KEY_SETUP.md for detailed instructions")
        return False
    
    print(f"\n✅ GOOGLE_API_KEY found: {api_key[:10]}...{api_key[-4:]}")
    print(f"   (Length: {len(api_key)} characters)")
    
    # Try to import google.adk
    print("\n📦 Checking google-adk installation...")
    try:
        from google.adk import Agent
        print("✅ google-adk is installed")
    except ImportError as e:
        print(f"❌ google-adk not installed: {e}")
        print("\nInstall with:")
        print("   pip install google-adk")
        return False
    
    # Try to create a test agent
    print("\n🤖 Testing Gemini connection...")
    try:
        import asyncio
        
        agent = Agent(
            model="gemini-2.5-flash",
            name="test_agent",
            instruction="You are a test assistant."
        )
        
        # Google ADK uses async methods
        async def test_agent():
            response = await agent.run_async("Say 'Hello from Gemini!' in exactly those words.")
            return response
        
        response = asyncio.run(test_agent())
        
        print(f"✅ Connection successful!")
        print(f"   Response: {str(response)[:100]}")
        
        print("\n" + "=" * 70)
        print("🎉 ALL CHECKS PASSED - You're ready to use LLM agents!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Extract drug names with LLM:")
        print("     python scripts/extract_drug_names.py --medications data/medications_synthetic.csv --use-llm")
        print("\n  2. Build ATC database:")
        print("     python scripts/build_atc_database.py --drug-names data/drug_names.txt")
        
        return True
        
    except Exception as e:
        print(f"❌ Error connecting to Gemini: {e}")
        print("\nPossible issues:")
        print("  - Invalid API key")
        print("  - Network connection problem")
        print("  - Rate limit exceeded")
        print("\nVerify your API key at: https://aistudio.google.com/apikey")
        return False


if __name__ == "__main__":
    success = check_api_key()
    sys.exit(0 if success else 1)
