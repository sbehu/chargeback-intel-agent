import os
import json
import pdfplumber

def format_table_to_markdown(table_matrix):
    """Converts a raw grid array of cells into a structured Markdown text table block."""
    if not table_matrix or not any(table_matrix):
        return ""
    
    cleaned_rows = []
    for row in table_matrix:
        if row and any(cell and str(cell).strip() for cell in row):
            cleaned_rows.append([str(cell).strip() if cell else "" for cell in row])
            
    if not cleaned_rows:
        return ""

    markdown_lines = []
    markdown_lines.append("| " + " | ".join(cleaned_rows[0]) + " |")
    markdown_lines.append("|" + "---| " * len(cleaned_rows[0]))
    
    for row in cleaned_rows[1:]:
        markdown_lines.append("| " + " | ".join(row) + " |")
        
    return "\n" + "\n".join(markdown_lines) + "\n"

def extract_high_value_rules(pdf_path, network_name):
    print(f"📊 Layout-Aware Scan + Context Enrichment for: {network_name}...")
    
    if not os.path.exists(pdf_path):
        print(f"❌ Error: Could not find manual file at {pdf_path}")
        return []

    parsed_chunks = []
    high_value_keywords = ["dispute", "chargeback", "reason code", "compelling evidence", "condition"]
    noise_keywords = ["glossary", "appendix", "table of contents", "password security", "cvm"]

    WINDOW_SIZE = 400   
    OVERLAP = 100       

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            raw_text = page.extract_text() or ""
            if not raw_text.strip() or "........" in raw_text or raw_text.count(".") > 15:
                continue
                
            cleaned_text = raw_text.lower()
            
            if any(keyword in cleaned_text for keyword in high_value_keywords):
                section_tag = "dispute_resolution"
                
                if any(noise in cleaned_text for noise in noise_keywords) and "condition" not in cleaned_text:
                    section_tag = "administrative_policies"

                # Extract and format any dense data grids on this page
                extracted_tables = page.extract_tables()
                table_string_block = ""
                for t in extracted_tables:
                    table_string_block += format_table_to_markdown(t)

                full_page_content = raw_text
                if table_string_block:
                    full_page_content += f"\n\n[STRUCTURED DATA TRANSCRIPT]:\n{table_string_block}"

                # Slice page data into windows
                words = full_page_content.split()
                for i in range(0, len(words), WINDOW_SIZE - OVERLAP):
                    window_words = words[i:i + WINDOW_SIZE]
                    raw_sliced_chunk = " ".join(window_words).strip()
                    
                    if len(raw_sliced_chunk) > 150:
                        # 🏷️ THE ENRICHMENT STEP: Inject context anchors directly into the text string
                        enriched_text_block = (
                            f"[DOCUMENT CONTEXT] Network: {network_name} | Section: {section_tag} | Page Reference: {page_num + 1}\n"
                            f"[RAW RULE EVIDENCE]:\n{raw_sliced_chunk}"
                        )
                        
                        parsed_chunks.append({
                            "text": enriched_text_block,  # 🌟 The math vector will now see these headers!
                            "metadata": {
                                "network": network_name,
                                "section": section_tag,
                                "page": page_num + 1,
                                "source_file": os.path.basename(pdf_path)
                            }
                        })
                        
    print(f"✅ Context Enrichment complete. Generated {len(parsed_chunks)} pristine chunks.")
    return parsed_chunks

if __name__ == "__main__":
    os.makedirs("raw_rulebooks", exist_ok=True)
    networks = {
        "Visa": "raw_rulebooks/visa_rulebook.pdf",
        "Mastercard": "raw_rulebooks/mastercard_rulebook.pdf",
        "Amex": "raw_rulebooks/amex_rulebook.pdf"
    }
    
    all_knowledge_chunks = []
    for network, path in networks.items():
        if os.path.exists(path):
            chunks = extract_high_value_rules(path, network)
            all_knowledge_chunks.extend(chunks)

    if all_knowledge_chunks:
        with open("raw_rulebooks/parsed_corpus.json", "w", encoding="utf-8") as f:
            json.dump(all_knowledge_chunks, f, indent=4)
        print(f"\n🎉 Success! Staged context-enriched chunks into 'raw_rulebooks/parsed_corpus.json'.")