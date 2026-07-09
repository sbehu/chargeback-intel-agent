import os
import json
import shutil
import random
import time
from datetime import datetime, timedelta
import pandas as pd

def safe_clear_directory(directory_path, max_retries=5, delay=0.5):
    """Safely clears a directory on Windows systems, handling temporary file locks."""
    if os.path.exists(directory_path):
        for attempt in range(max_retries):
            try:
                shutil.rmtree(directory_path)
                break
            except PermissionError:
                if attempt < max_retries - 1:
                    time.sleep(delay)
                else:
                    print(f"⚠️ Warning: Could not clear directory {directory_path} due to a Windows file lock.")
                    raise

def generate_random_street():
    """Generates an organic street address name."""
    names = ["Greenwood", "Oak", "Pine", "Maple", "Cedar", "Washington", "Lincoln", "Heritage", "Summit", "Hillside", "River", "Sunset"]
    types = ["St", "Ave", "Blvd", "Dr", "Ln", "Ct", "Way"]
    return f"{random.randint(100, 9999)} {random.choice(names)} {random.choice(types)}"

def generate_random_location():
    """Generates a completely randomized State and ZIP pair for anomalies."""
    states = ["CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI", "WA", "AZ"]
    random_zip = f"{random.randint(10000, 99999)}"
    return random.choice(states), random_zip

def run_ingestion():
    csv_path = "data/complaints.csv"
    bucket_dir = "mock_s3_bucket/cases"
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: Baseline source file not found at {csv_path}")
        return

    print("Reading source database...")
    df = pd.read_csv(csv_path)
    
    safe_clear_directory(bucket_dir)
    os.makedirs(bucket_dir, exist_ok=True)
    
    df_clean = df[df["Consumer complaint narrative"].notna()].head(10000)
    print(f"Loaded {len(df_clean)} records. Injecting timeline and address mismatch matrix...")

    for idx, row in df_clean.iterrows():
        case_id = f"dispute_{idx}"
        case_folder = os.path.join(bucket_dir, case_id)
        os.makedirs(case_folder, exist_ok=True)
        
        # --- 1. CORE DISPUTE ANCHOR DATE & DATA QUALITY CHECK ---
        raw_date_str = row.get("Date received")
        data_quality_flag = "HEALTHY"
        
        if pd.isna(raw_date_str) or str(raw_date_str).strip() == "":
            raw_date_str = "2026-04-15"
            data_quality_flag = "MISSING_SOURCE_DATE"
            
        try:
            dispute_received_date = datetime.strptime(str(raw_date_str), "%Y-%m-%d")
        except ValueError:
            dispute_received_date = datetime.strptime("2026-04-15", "%Y-%m-%d")
            data_quality_flag = "CORRUPT_SOURCE_DATE"
            
        # --- 2. EXTRACT TRUE COMPLAINT LOCATION ---
        complaint_state = str(row.get("State", "NY")).upper()
        complaint_zip = str(row.get("ZIP code", "10001")).split(".")[0]
        
        # --- 3. ADDRESS MISMATCH / RISK MATRIX SELECTION ---
        address_roll = random.random()
        
        if address_roll < 0.05:
            # Scenario A: Customer Moved / Gift Profile (Invoice matches Complaint, Shipping Mismatches)
            invoice_state, invoice_zip = complaint_state, complaint_zip
            shipping_state, shipping_zip = generate_random_location()
            address_profile = "CUSTOMER_MOVE_OR_GIFT"
        elif address_roll >= 0.05 and address_roll < 0.08:
            # Scenario B: True Identity Theft / Fraud (Invoice & Shipping match each other, but clash with Victim)
            fraud_state, fraud_zip = generate_random_location()
            invoice_state, invoice_zip = fraud_state, fraud_zip
            shipping_state, shipping_zip = fraud_state, fraud_zip
            address_profile = "IDENTITY_THEFT_FRAUD"
        else:
            # Standard Scenario: Happy Path (All locations align perfectly)
            invoice_state, invoice_zip = complaint_state, complaint_zip
            shipping_state, shipping_zip = complaint_state, complaint_zip
            address_profile = "MATCHED_LOCATION"

        # Build fully localized string profiles
        invoice_address = f"{generate_random_street()}, {invoice_state}, {invoice_zip}"
        shipping_address = f"{generate_random_street()}, {shipping_state}, {shipping_zip}"
        
        # --- 4. MULTI-SCENARIO TIMELINE ASSIGNMENT ---
        dice_roll = random.random()
        has_delivery_receipt = True
        
        if dice_roll < 0.80:
            days_back_for_invoice = random.randint(30, 45)
            days_to_ship = random.randint(1, 2)
            days_to_deliver = random.randint(2, 7)
            tracking_status = "DELIVERED_ON_TIME"
        elif dice_roll >= 0.80 and dice_roll < 0.85:
            days_back_for_invoice = random.randint(150, 180) # Triggers Policy Agent Expired State
            days_to_ship = random.randint(1, 2)
            days_to_deliver = random.randint(2, 7)
            tracking_status = "DELIVERED_ON_TIME"
        elif dice_roll >= 0.85 and dice_roll < 0.88:
            days_back_for_invoice = random.randint(45, 60)
            days_to_ship = random.randint(15, 30)
            days_to_deliver = random.randint(2, 7)
            tracking_status = "DELIVERED_DELAYED_FULFILLMENT"
        elif dice_roll >= 0.88 and dice_roll < 0.91:
            days_back_for_invoice = random.randint(45, 60)
            days_to_ship = random.randint(1, 2)
            days_to_deliver = random.randint(14, 25)
            tracking_status = "DELIVERED_DELAYED_TRANSIT"
        elif dice_roll >= 0.91 and dice_roll < 0.94:
            days_back_for_invoice = random.randint(60, 90)
            days_to_ship = random.randint(15, 30)
            days_to_deliver = random.randint(14, 25)
            tracking_status = "DELIVERED_COMPOUND_DELAY"
        elif dice_roll >= 0.94 and dice_roll < 0.97:
            days_back_for_invoice = random.randint(30, 45)
            days_to_ship = random.randint(1, 2)
            days_to_deliver = random.randint(30, 40)
            tracking_status = "LOST_IN_TRANSIT"
        else:
            days_back_for_invoice = random.randint(30, 45)
            days_to_ship = None
            days_to_deliver = None
            tracking_status = "UNFULFILLED_GHOST_ORDER"
            has_delivery_receipt = False

        invoice_date = dispute_received_date - timedelta(days=days_back_for_invoice)
        
        # --- 5. WRITE ARTIFACT: METADATA.JSON ---
        metadata = {
            "case_id": case_id,
            "bank_provider": str(row.get("Company", "UNKNOWN_BANK")).upper(),
            "card_network": random.choice(["visa", "mastercard", "amex"]),
            "reason_code": random.choice(["10.4", "UA01", "F29"]),
            "data_quality_audit": data_quality_flag,
            "risk_profile": {
                "address_anomaly_type": address_profile,
                "complaint_state": complaint_state,
                "complaint_zip": complaint_zip
            },
            "timeline": {
                "transaction_date": invoice_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "dispute_received_date": dispute_received_date.strftime("%Y-%m-%dT%H:%M:%SZ")
            },
            "transaction_details": {
                "amount_cents": random.randint(500, 50000),
                "currency": "USD"
            },
            "raw_consumer_statement": row["Consumer complaint narrative"]
        }
        
        with open(os.path.join(case_folder, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # --- 6. WRITE ARTIFACT: INVOICE.JSON ---
        invoice_payload = {
            "invoice_id": f"INV-{random.randint(100000, 999999)}",
            "invoice_date": invoice_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "amount_cents": metadata["transaction_details"]["amount_cents"],
            "currency": "USD",
            "billing_address": invoice_address
        }
        with open(os.path.join(case_folder, "invoice.json"), "w", encoding="utf-8") as f:
            json.dump(invoice_payload, f, indent=2)

        # --- 7. WRITE ARTIFACT: FEDEX_DELIVERY_RECEIPT.TXT ---
        if has_delivery_receipt:
            ship_date = invoice_date + timedelta(days=days_to_ship)
            delivery_date = ship_date + timedelta(days=days_to_deliver)
            
            receipt_text = f"""============================================================
                    FEDEX DELIVERY RECEIPT                  
============================================================
Tracking Number: FX-{random.randint(10000000, 99999999)}-US
Shipment Date:   {ship_date.strftime("%Y-%m-%dT%H:%M:%SZ")}
Delivery Date:   {delivery_date.strftime("%Y-%m-%dT%H:%M:%SZ")}
Status:          {tracking_status.upper()} / SIGNED BY CUSTOMER
Destination:     {shipping_address.upper()}
============================================================
"""
            with open(os.path.join(case_folder, "fedex_delivery_receipt.txt"), "w", encoding="utf-8") as f:
                f.write(receipt_text)

    print("🎉 SUCCESS! The definitive high-fidelity risk-modeling testing engine is fully built.")

if __name__ == "__main__":
    run_ingestion()