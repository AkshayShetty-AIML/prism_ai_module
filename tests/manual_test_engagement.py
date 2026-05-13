import os
import sys
import json

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
# Load real API keys for the test
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from pipeline.engagement_generator import generate_engagement

# ---------------------------------------------------------
# Mock Database Implementation (Dependency Injection)
# ---------------------------------------------------------
class MockCollection:
    def __init__(self, data):
        self.data = data
        
    def find(self, query, projection=None):
        # We ignore the query and projection for the mock
        # and just return the dataset
        return self
        
    def limit(self, count):
        return self.data[:count]

class MockDB:
    def __init__(self):
        # 3 mock comments representing a negative theme group
        self.analyzed_records = MockCollection([
            {
                "platform": "youtube", 
                "content": "Second half is so lagging. I almost slept.", 
                "sentiment": "negative"
            },
            {
                "platform": "twitter", 
                "content": "Pacing issue in the 2nd half. Director failed to maintain the thrill. Overhyped.", 
                "sentiment": "negative"
            },
            {
                "platform": "reddit", 
                "content": "Why was the flashback so slow? Ruined the whole movie pacing.", 
                "sentiment": "negative"
            }
        ])

# ---------------------------------------------------------
# Test Runner
# ---------------------------------------------------------
def run_test():
    provider = os.getenv("LLM_PROVIDER", "groq")
    print(f"\n{'='*70}")
    print(f"PRISM Engagement Generator — Manual Test (Provider: {provider.upper()})")
    print(f"{'='*70}")
    
    db = MockDB()
    print("⏳ Fetching mock comments & calling LLM...")
    
    # Call the actual pipeline function, injecting our Mock DB
    result = generate_engagement(
        theme_group_id="mock-cluster-123",
        keyword="Leo movie",
        db=db
    )
    
    print("\n📊 Final Dashboard Payload:")
    print(json.dumps(result, indent=2))
    
    print(f"\n{'='*70}")
    if "error" in result:
        print("❌ FAILED — Encountered an error")
    elif len(result.get("suggested_replies", [])) >= 3:
        print("✅ PASSED — Generated 3+ replies in different tones")
    else:
        print("⚠️ WARNING — Generated fewer than 3 replies")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    run_test()
