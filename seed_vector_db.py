import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

def get_env_var(var_name):
    val = os.environ.get(var_name)
    if not val and os.path.exists(".env"):
        with open(".env", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(f"{var_name}="):
                    return line.split("=", 1)[1].strip()
    return val

# Verify cloud credentials safely
openai_key = get_env_var("OPENAI_API_KEY")
pinecone_key = get_env_var("PINECONE_API_KEY")

if not openai_key or not pinecone_key:
    raise ValueError("❌ Missing API keys in environment configuration.")

# Initialize API clients
openai_client = OpenAI(api_key=openai_key)
pc = Pinecone(api_key=pinecone_key)

INDEX_NAME = "chargeback-rules"
EMBEDDING_MODEL = "text-embedding-3-small"

# Verify or create clean Pinecone index
if INDEX_NAME not in pc.list_indexes().names():
    print(f"📡 Creating fresh cloud vector index: '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)


def clear_existing_index():
    """
    Deletes all existing vectors in the index before re-seeding with the new
    header-aware chunks. This is DESTRUCTIVE - it wipes whatever is currently
    live in production. Without this, old fixed-window chunks (from the
    previous parsing scheme) would remain alongside the new ones, since
    chunk IDs are just sequential integers and a re-run with a different
    chunk count won't overwrite every old ID.
    """
    print(f"⚠️  Clearing all existing vectors in index '{INDEX_NAME}' before re-seeding...")
    index.delete(delete_all=True)
    print("✅ Index cleared.")


def seed_database():
    corpus_path = "raw_rulebooks/parsed_corpus.json"
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"❌ Staging file missing at {corpus_path}. Run parser first.")

    with open(corpus_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"🚀 Found {len(chunks)} staged blocks. Preparing cloud deployment payloads...")

    # Process in batches of 25 to respect network payloads safely
    BATCH_SIZE = 25
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        upsert_payload = []

        for idx, item in enumerate(batch):
            global_id = f"chunk_{i + idx}"
            raw_text = item["text"]
            source_metadata = item["metadata"]

            # Generate mathematical vector coordinates from OpenAI
            response = openai_client.embeddings.create(
                input=[raw_text],
                model=EMBEDDING_MODEL
            )
            vector_coords = response.data[0].embedding

            # Combine text content and our hierarchical drawer tags into the metadata pack
            metadata_payload = {
                "text": raw_text,
                "network": source_metadata["network"],
                "section": source_metadata["section"],
                "page": source_metadata["page"],
                "condition_id": source_metadata.get("condition_id", "unspecified"),
            }

            upsert_payload.append((global_id, vector_coords, metadata_payload))

        # Write directly to AWS cloud cluster nodes
        index.upsert(vectors=upsert_payload)
        print(f"📦 Successfully synced batch {i // BATCH_SIZE + 1} ({len(upsert_payload)} vectors) to cloud index.")

    print("\n🎉 Pinecone Database completely re-indexed with header-aware condition metadata!")


if __name__ == "__main__":
    confirm = input(
        "This will DELETE all existing vectors in the 'chargeback-rules' Pinecone "
        "index and replace them with newly parsed, header-aware chunks. This "
        "cannot be undone. Type 'yes' to continue: "
    )
    if confirm.strip().lower() == "yes":
        clear_existing_index()
        seed_database()
    else:
        print("Aborted. No changes made to the index.")