import os
import json
from agents import PolicyComplianceAgent, LogisticsComplianceAgent

def test_pipeline():
    policy_agent = PolicyComplianceAgent()
    logistics_agent = LogisticsComplianceAgent()
    cases_dir = "mock_s3_bucket/cases"
    
    if not os.path.exists(cases_dir):
        print(f"❌ Error: Case directory {cases_dir} does not exist. Run ingestion.py first!")
        return

    print("🚀 Executing Combined Policy & Fraud Audit Scan (First 150 Cases)...")
    print("-" * 85)

    all_cases = os.listdir(cases_dir)[:150]
    
    counters = {
        "EXPIRED": 0, "MATCHED": 0, "GIFT": 0, "FRAUD": 0, "LOST": 0, "GHOST": 0
    }

    for case_id in all_cases:
        case_path = os.path.join(cases_dir, case_id)
        metadata_path = os.path.join(case_path, "metadata.json")
        
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            
        # 1. First Pass: Run Policy Check Gatekeeper
        policy_result = policy_agent.check_timeline(metadata_path)
        
        if policy_result["status"] == "SUCCESS" and policy_result["verdict"] == "EXPIRED_DISPUTE_WINDOW":
            counters["EXPIRED"] += 1
            print(f"🛑 [POLICY BREACH] {case_id}: Expired filing window ({policy_result['days_elapsed']} days elapsed).")
            continue # Technical knockout win. Move immediately to next file.

        # 2. Second Pass: Run Logistics Fraud Auditor
        logistics_result = logistics_agent.analyze_fulfillment(case_path, metadata)
        state = logistics_result["fulfillment_state"]
        geo = logistics_result.get("geo_profile", "UNFULFILLED")
        
        if state == "UNFULFILLED_GHOST_ORDER":
            counters["GHOST"] += 1
            print(f"👻 [GHOST ORDER]   {case_id}: No shipping record! Merchant infrastructure breakdown.")
        elif state == "LOST_IN_TRANSIT":
            counters["LOST"] += 1
            print(f"📦 [LOST IN TRANSIT] {case_id}: Courier dropped the ball. Instant merchant liability.")
        elif geo == "IDENTITY_THEFT_FRAUD":
            counters["FRAUD"] += 1
            print(f"🚨 [CRITICAL FRAUD] {case_id}: Identity theft! Shipped to outside criminal hub.")
        elif geo == "CUSTOMER_MOVE_OR_GIFT":
            counters["GIFT"] += 1
            print(f"🎁 [LOCATION CHANGE] {case_id}: Genuine mismatch. Shipped as gift/relocation.")
        else:
            counters["MATCHED"] += 1

    print("-" * 85)
    print("📋 AUDIT PROFILE SUMMARY COMPLETED:")
    for key, count in counters.items():
        print(f"   • {key.upper()}: {count} cases trapped.")

if __name__ == "__main__":
    test_pipeline()