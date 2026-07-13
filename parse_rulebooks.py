import os
import re
import json
import pdfplumber

# Matches headers like "Dispute Condition 13.5: Misrepresentation" or
# "Dispute Condition 10.3 Other Fraud - Card Present Environment" - tolerant
# of missing colon and irregular spacing from PDF text extraction.
CONDITION_HEADER_PATTERN = re.compile(
    r"Dispute Condition\s+(\d+(?:\.\d+)?)\s*:?\s*([A-Z][A-Za-z /\-]{2,60})?"
)

WINDOW_SIZE = 400
OVERLAP = 100


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


def split_into_condition_segments(page_text):
    """
    Splits raw page text into segments anchored at 'Dispute Condition X.X' headers,
    so unrelated conditions/tables that happen to sit near each other on the PDF
    page are never fused into the same retrieval chunk. Any text before the first
    detected header (or on pages with no header at all) becomes a single
    'unlabeled' segment, tagged with condition_id=None, and still gets word-window
    chunked below like before - it just won't get mislabeled with a header it
    doesn't actually belong to.

    Returns a list of dicts: {"condition_id": str or None, "text": str}
    """
    matches = list(CONDITION_HEADER_PATTERN.finditer(page_text))

    if not matches:
        return [{"condition_id": None, "text": page_text}]

    segments = []

    # Preamble before the first header, if any
    if matches[0].start() > 0:
        preamble = page_text[: matches[0].start()].strip()
        if preamble:
            segments.append({"condition_id": None, "text": preamble})

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(page_text)
        segment_text = page_text[start:end].strip()
        condition_id = match.group(1)
        if segment_text:
            segments.append({"condition_id": condition_id, "text": segment_text})

    return segments


def window_chunk_text(text, window_size=WINDOW_SIZE, overlap=OVERLAP):
    """Splits text into overlapping word-count windows. Used within a single
    condition segment (never across a header boundary), so a long condition's
    text still gets broken into retrievable chunks without bleeding into a
    neighboring, unrelated condition."""
    words = text.split()
    if len(words) <= window_size:
        return [text]

    windows = []
    for i in range(0, len(words), window_size - overlap):
        window_words = words[i: i + window_size]
        windows.append(" ".join(window_words))
    return windows


def extract_high_value_rules(pdf_path, network_name):
    print(f"📊 Header-Aware Segmented Scan for: {network_name}...")

    if not os.path.exists(pdf_path):
        print(f"❌ Error: Could not find manual file at {pdf_path}")
        return []

    parsed_chunks = []
    high_value_keywords = ["dispute", "chargeback", "reason code", "compelling evidence", "condition"]
    noise_keywords = ["glossary", "appendix", "table of contents", "password security", "cvm"]

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            raw_text = page.extract_text() or ""
            if not raw_text.strip() or "........" in raw_text or raw_text.count(".") > 15:
                continue

            cleaned_text_lower = raw_text.lower()

            if not any(keyword in cleaned_text_lower for keyword in high_value_keywords):
                continue

            section_tag = "dispute_resolution"
            if any(noise in cleaned_text_lower for noise in noise_keywords) and "condition" not in cleaned_text_lower:
                section_tag = "administrative_policies"

            # --- Header-anchored text segmentation (the actual fix) ---
            condition_segments = split_into_condition_segments(raw_text)

            for segment in condition_segments:
                condition_id = segment["condition_id"]
                for window_text in window_chunk_text(segment["text"]):
                    if len(window_text) <= 150:
                        continue

                    condition_line = f"Condition Reference: {condition_id}\n" if condition_id else ""
                    enriched_text_block = (
                        f"[DOCUMENT CONTEXT] Network: {network_name} | Section: {section_tag} | "
                        f"Page Reference: {page_num + 1}\n"
                        f"{condition_line}"
                        f"[RAW RULE EVIDENCE]:\n{window_text}"
                    )

                    parsed_chunks.append({
                        "text": enriched_text_block,
                        "metadata": {
                            "network": network_name,
                            "section": section_tag,
                            "page": page_num + 1,
                            "condition_id": condition_id if condition_id else "unspecified",
                            "source_file": os.path.basename(pdf_path)
                        }
                    })

            # --- Tables: kept as their own standalone chunk(s), never fused
            # into a text window. If the page has exactly one detected
            # condition header, we tag the table with that condition_id as a
            # reasonable heuristic (the table almost certainly belongs to
            # that one condition). If there are zero or multiple headers on
            # the page, we leave it "unspecified" rather than guess wrong -
            # pdfplumber doesn't give us reliable positional ordering between
            # extracted tables and surrounding text to attribute this more
            # precisely without deeper layout analysis. ---
            headered_ids = [seg["condition_id"] for seg in condition_segments if seg["condition_id"]]
            table_condition_id = headered_ids[0] if len(headered_ids) == 1 else "unspecified"

            extracted_tables = page.extract_tables()
            for t in extracted_tables:
                table_markdown = format_table_to_markdown(t)
                if len(table_markdown.strip()) <= 20:
                    continue

                condition_line = f"Condition Reference: {table_condition_id}\n" if table_condition_id != "unspecified" else ""
                enriched_table_block = (
                    f"[DOCUMENT CONTEXT] Network: {network_name} | Section: {section_tag} | "
                    f"Page Reference: {page_num + 1}\n"
                    f"{condition_line}"
                    f"[STRUCTURED DATA TRANSCRIPT]:\n{table_markdown}"
                )

                parsed_chunks.append({
                    "text": enriched_table_block,
                    "metadata": {
                        "network": network_name,
                        "section": section_tag,
                        "page": page_num + 1,
                        "condition_id": table_condition_id,
                        "source_file": os.path.basename(pdf_path)
                    }
                })

    print(f"✅ Segmented parsing complete. Generated {len(parsed_chunks)} header-aware chunks.")
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
        print(f"\n🎉 Success! Staged {len(all_knowledge_chunks)} header-aware chunks into 'raw_rulebooks/parsed_corpus.json'.")