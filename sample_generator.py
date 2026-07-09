import os
import json
import pandas as pd

def generate_golden_test_bench():
    csv_path = os.path.join("data", "complaints.csv")
    output_json_path = "test_bench.json"
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ Cannot find raw data file at: {csv_path}")
        
    print(f"[+] Reading raw data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    TEXT_COLUMN = "Consumer complaint narrative"
    CATEGORY_COLUMN = "Issue"

    # Step 1: Drop rows that do not have a customer narrative text
    print("[+] Filtering out empty narratives...")
    df = df.dropna(subset=[TEXT_COLUMN])
    total_valid_rows = len(df)
    print(f"[+] Found {total_valid_rows} rows with valid text narratives.")
    
    TARGET_SIZE = 400
    
    # Step 2: Stratified sampling safely using dynamic value counts
    print(f"[+] Calculating stratified category distributions...")
    proportions = df[CATEGORY_COLUMN].value_counts(normalize=True)
    
    stratified_rows = []
    
    for category, prop in proportions.items():
        # Determine exactly how many samples this specific issue needs
        sample_n = int(round(prop * TARGET_SIZE))
        if sample_n == 0:
            continue
            
        # Extract matches safely
        category_pool = df[df[CATEGORY_COLUMN] == category]
        
        # Pull the samples securely
        sampled_subset = category_pool.sample(
            n=min(sample_n, len(category_pool)),
            random_state=42
        )
        stratified_rows.append(sampled_subset)
        
    # Combine the subsets and clamp to exactly 400
    stratified_sample = pd.concat(stratified_rows).head(TARGET_SIZE)
    
    # Step 3: Map data fields into your clean json structure
    json_ready_data = []
    for idx, row in stratified_sample.iterrows():
        category = str(row[CATEGORY_COLUMN])
        narrative_text = str(row[TEXT_COLUMN])
        
        if "fraud" in category.lower() or "unauthorized" in category.lower() or "stolen" in category.lower():
            expected_verdict = "CHALLENGE"
        else:
            expected_verdict = "ACCEPT"
            
        json_ready_data.append({
            "case_id": f"EVAL-{1000 + len(json_ready_data) + 1}",
            "dispute_type": category,
            "expected_verdict": expected_verdict,
            "narrative": narrative_text
        })
        
    # Step 4: Write to test_bench.json
    with open(output_json_path, "w") as f:
        json.dump(json_ready_data, f, indent=4)
        
    print("\n" + "="*60)
    print("✅ SUCCESS: FIXED STRATIFIED GENERATION COMPLETE")
    print("="*60)
    print(f" • Sample Target Size    : {len(json_ready_data)} cases successfully processed.")
    print(f" • Overwritten File Path : {output_json_path}")
    print("="*60 + "\n")

if __name__ == "__main__":
    generate_golden_test_bench()