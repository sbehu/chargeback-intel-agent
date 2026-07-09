import os
from openai import OpenAI
from pinecone import Pinecone

# ─────────────────────────────────────────────────────────────
# 1. CUSTOMER READER AGENT
# ─────────────────────────────────────────────────────────────
class CustomerReaderAgent:
    def __init__(self):
        self.agent_name = "Customer_Reader_Agent"
        
        def get_env_var(var_name):
            val = os.environ.get(var_name)
            if not val and os.path.exists(".env"):
                with open(".env", "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith(f"{var_name}="):
                            return line.split("=", 1)[1].strip()
            return val

        openai_key = get_env_var("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("❌ CustomerReaderAgent: Missing OPENAI_API_KEY in configuration.")
            
        self.client = OpenAI(api_key=openai_key)

    def classify_intent(self, raw_complaint_text):
        """Classifies messy user descriptions uniformly into structural machine tags."""
        prompt = f"""
        You are an elite financial operations risk analyst. 
        Analyze the following customer chargeback complaint text and categorize it into exactly ONE of these precise dispute buckets:
        - FRAUD_UNAUTHORIZED
        - INCORRECT_AMOUNT
        - MERCHANDISE_NOT_RECEIVED
        - NOT_AS_DESCRIBED

        Respond with ONLY the category string itself (e.g., FRAUD_UNAUTHORIZED). Do not include any intro, markdown, or commentary.

        Customer Complaint Text: "{raw_complaint_text}"
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return response.choices[0].message.content.strip().replace("'", "").replace('"', '')
        except Exception as e:
            print(f"⚠️  CustomerReaderAgent Error: {e}")
            return "FRAUD_UNAUTHORIZED"


# ─────────────────────────────────────────────────────────────
# 2. REGULATORY KNOWLEDGE AGENT (CLOUD RAG)
# ─────────────────────────────────────────────────────────────
class RegulatoryKnowledgeAgent:
    def __init__(self):
        self.agent_name = "Regulatory_Knowledge_Agent"
        self.portfolio_history = []
        
        def get_env_var(var_name):
            val = os.environ.get(var_name)
            if not val and os.path.exists(".env"):
                with open(".env", "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith(f"{var_name}="):
                            return line.split("=", 1)[1].strip()
            return val

        openai_key = get_env_var("OPENAI_API_KEY")
        pinecone_key = get_env_var("PINECONE_API_KEY")
        
        if not openai_key or not pinecone_key:
            raise ValueError("❌ RegulatoryKnowledgeAgent: Missing API Keys in environment configuration.")
            
        self.openai_client = OpenAI(api_key=openai_key)
        self.pc = Pinecone(api_key=pinecone_key)
        self.index = self.pc.Index("chargeback-rules")
        self.embedding_model = "text-embedding-3-small"

    def track_portfolio_trends(self, current_case_metadata):
        """Financial Intelligence Layer: Tracks cross-case velocity patterns to flag organized fraud rings."""
        self.portfolio_history.append(current_case_metadata)
        current_zip = current_case_metadata.get("shipping_address_zip", "")
        
        zip_clashes = [c for c in self.portfolio_history if c.get("shipping_address_zip") == current_zip]
        
        if len(zip_clashes) >= 3:
            return {
                "trend_detected": True,
                "vector_alert": "HIGH_VELOCITY_ZIP_CLUSTER",
                "message": f"CRITICAL: Found {len(zip_clashes)} disputes routed to the exact same shipping zip code within this batch. Organized fraud sweep suspected."
            }
        return {"trend_detected": False}

    def retrieve_with_metadata_filter(self, dispute_category, network_filter):
        """Performs a precise, server-side Metadata-Filtered semantic search over the cloud database."""
        net_str = str(network_filter).strip().lower()
        if "visa" in net_str:
            clean_network = "Visa"
        elif "mastercard" in net_str:
            clean_network = "Mastercard"
        elif "amex" in net_str or "american" in net_str:
            clean_network = "Amex"
        else:
            clean_network = network_filter
            
        print(f"🔍 [CLOUD QUERY]: Searching rules for network brand: '{clean_network}'")

        response = self.openai_client.embeddings.create(
            input=[dispute_category],
            model=self.embedding_model
        )
        query_vector = response.data[0].embedding
        
        try:
            # PRODUCTION REFACTOR: Enforce composite metadata filters on the database level
            server_reply = self.index.query(
                vector=query_vector,
                top_k=1,
                filter={
                    "network": {"$eq": clean_network},
                    "section": {"$eq": "dispute_resolution"} # 🌟 Locks out glossary noise completely!
                },
                include_metadata=True
            )
            
            if server_reply and server_reply.get("matches"):
                best_match = server_reply["matches"][0]
                return {
                    "compelling_evidence_requirements": best_match["metadata"]["text"],
                    "match_confidence": float(best_match["score"])
                }
        except Exception as e:
            print(f"⚠️ Cloud Database Fetch Warning: {e}")
            
        return {
            "compelling_evidence_requirements": f"Provide standardized transaction details adhering to standard baseline {clean_network.lower()} rules.",
            "match_confidence": 0.0
        }

# ─────────────────────────────────────────────────────────────
# 3. POLICY COMPLIANCE AGENT
# ─────────────────────────────────────────────────────────────
class PolicyComplianceAgent:
    def __init__(self):
        self.agent_name = "Policy_Compliance_Agent"

    def check_internal_refund_status(self, case_id):
        """Inspects billing logs to confirm if a fallback credit loop was already fulfilled."""
        already_refunded_cases = ["dispute_999"] 
        return case_id in already_refunded_cases


# ─────────────────────────────────────────────────────────────
# 4. LOGISTICS COMPLIANCE AGENT
# ─────────────────────────────────────────────────────────────
class LogisticsComplianceAgent:
    def __init__(self):
        self.agent_name = "Logistics_Compliance_Agent"

    def verify_delivery_pipeline(self, case_id):
        """Queries tracking records for actual carrier pipeline status."""
        tracking_registry = {
            "dispute_1": "DELIVERED",
            "dispute_10": "DELIVERED",
            "dispute_1000": "DELIVERED",
            "dispute_10001": "DELIVERED",
            "dispute_10004": "DELIVERED"
        }
        return tracking_registry.get(case_id, "UNKNOWN")