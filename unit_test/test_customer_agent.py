import os
import json
from agents import CustomerReaderAgent

def test_ai_reader():
    reader_agent = CustomerReaderAgent()
    cases_dir = "mock_s3_bucket/cases"
    
    if not os.path.exists(cases_dir):
        print(f"❌ Error: Case directory {cases_dir} does not exist. Run ingestion.py first!")
        return

    print("🚀 Running Customer Reader Agent Live Test (First 5 Narrative Extractions)...")
    print("=" * 90)

    # Pick the first 5 cases to see the AI analysis live
    all_cases = os.listdir(cases_dir)[:5]

    for case_id in all_cases:
        metadata_path = os.path.join(cases_dir, case_id, "metadata.json")
        
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            
        raw_text = metadata.get("raw_consumer_statement", "")
        
        print(f"🔍 [CASE FILING]: {case_id}")
        print(f"📄 [RAW TEXT EXTRACT]:\n{raw_text[:200]}...") # Show snippet of the long text
        
        # Execute the LLM analysis pass
        ai_result = reader_agent.analyze_narrative(raw_text)
        
        if ai_result["status"] == "SUCCESS":
            print(f"🏷️  [AI CLASSIFICATION]: {ai_result['category']} (Confidence: {ai_result['confidence']})")
            print(f"📝 [AI ARGUMENT SUMMARY]: {ai_result['summary']}")
        else:
            print(f"❌ [AI ERROR]: {ai_result['reason']}")
            
        print("-" * 90)

if __name__ == "__main__":
    test_ai_reader()