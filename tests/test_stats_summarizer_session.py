"""
Test Stats Summarizer Agent with Session-Based Pattern

This demonstrates the Day 3a pattern: stateful, interactive analysis
with Code Execution for pandas/matplotlib.
"""

import pytest
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import asyncio

# Load .env file for API key
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.stats_summarizer import (
    create_stats_session,
    query_stats_session,
    close_stats_session
)


# Fixture to check for API key
@pytest.fixture(scope="session")
def api_key_available():
    """Check if GOOGLE_API_KEY is available."""
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        pytest.skip("GOOGLE_API_KEY not found. Set it in .env file or environment variable.")
    return api_key


class TestStatsSessionAgent:
    """Test the session-based Stats Summarizer agent."""
    
    @pytest.mark.asyncio
    async def test_create_and_query_session(self, api_key_available):
        """
        Test creating a session and running interactive queries.
        
        This demonstrates:
        1. Creating a session with data files
        2. Asking multi-turn questions
        3. Getting tables and statistics back
        """
        # Use test data
        meds_file = "tests/input1/medications_test.csv"
        diag_file = "tests/input1/conditions_test.csv"
        
        if not Path(meds_file).exists() or not Path(diag_file).exists():
            pytest.skip("Test data files not found")
        
        # Create session with data
        session_id = await create_stats_session(
            medications_file=meds_file,
            diagnoses_file=diag_file
        )
        
        print(f"\n✅ Created session: {session_id}")
        
        try:
            # Query 1: Basic data summary
            response1 = await query_stats_session(
                session_id,
                "What data has been loaded? Give me a summary."
            )
            print("\n" + "="*80)
            print("QUERY 1: Data Summary")
            print("="*80)
            print(response1)
            
            # Query 2: Drug analysis
            response2 = await query_stats_session(
                session_id,
                "Show me a table of the top 10 most frequently prescribed drugs. Include patient count and total prescriptions."
            )
            print("\n" + "="*80)
            print("QUERY 2: Top 10 Drugs")
            print("="*80)
            print(response2)
            
            # Query 3: Patient statistics
            response3 = await query_stats_session(
                session_id,
                "How many unique patients are in the medications data?"
            )
            print("\n" + "="*80)
            print("QUERY 3: Patient Count")
            print("="*80)
            print(response3)
            
            # Assertions
            assert session_id is not None
            assert len(response1) > 0
            assert len(response2) > 0
            assert len(response3) > 0
            
        finally:
            # Clean up session
            await close_stats_session(session_id)
            print(f"\n✅ Closed session: {session_id}")
    
    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, api_key_available):
        """
        Test multi-turn conversation with follow-up questions.
        
        This shows the power of stateful sessions - the agent
        remembers context from previous queries.
        """
        meds_file = "tests/input1/medications_test.csv"
        
        if not Path(meds_file).exists():
            pytest.skip("Test data file not found")
        
        session_id = await create_stats_session(medications_file=meds_file)
        
        try:
            # Turn 1: Ask about drugs
            response1 = await query_stats_session(
                session_id,
                "What are the most common drug descriptions in the data?"
            )
            print("\n" + "="*80)
            print("TURN 1: Most common drugs")
            print("="*80)
            print(response1)
            
            # Turn 2: Follow-up question (agent remembers context)
            response2 = await query_stats_session(
                session_id,
                "Now group those by patient and show me patients with the most medications."
            )
            print("\n" + "="*80)
            print("TURN 2: Patients with most meds")
            print("="*80)
            print(response2)
            
            assert len(response1) > 0
            assert len(response2) > 0
            
        finally:
            await close_stats_session(session_id)


if __name__ == "__main__":
    # Run the test directly (for development)
    async def main():
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            print("❌ GOOGLE_API_KEY not set")
            return
        
        print("🧪 Testing Stats Summarizer Session Agent")
        print("="*80)
        
        meds_file = "tests/input1/medications_test.csv"
        diag_file = "tests/input1/conditions_test.csv"
        
        session_id = await create_stats_session(
            medications_file=meds_file,
            diagnoses_file=diag_file
        )
        
        print(f"Session created: {session_id}\n")
        
        # Interactive example
        queries = [
            "Summarize the loaded data",
            "Show me the top 5 most prescribed drugs",
            "How many patients are in the dataset?"
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"\n{'='*80}")
            print(f"QUERY {i}: {query}")
            print('='*80)
            response = await query_stats_session(session_id, query)
            print(response)
        
        await close_stats_session(session_id)
        print(f"\n✅ Session closed")
    
    asyncio.run(main())
