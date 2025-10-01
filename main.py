import os
import re
import json
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama


# ---------------- PDF STRUCTURE EXTRACTION ---------------- #

section_pattern = re.compile(r'^\d+\.\s+.+')         # Example: "1. Introduction"
subsection_pattern = re.compile(r'^\d+\.\d+\s+.+')   # Example: "1.1 Background"

def extract_pdf_structure(pdf_path):
    doc = fitz.open(pdf_path)

    pdf_data = {
        "pdf_name": os.path.basename(pdf_path),
        "sections": []
    }

    current_section = None
    current_subsection = None

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        lines = text.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if section_pattern.match(line) and not subsection_pattern.match(line):
                current_section = {
                    "section_name": line,
                    "page_number": page_num,
                    "text": "",
                    "subsections": []
                }
                pdf_data["sections"].append(current_section)
                current_subsection = None

            elif subsection_pattern.match(line):
                current_subsection = {
                    "subsection_name": line,
                    "page_number": page_num,
                    "text": ""
                }
                if current_section is None:
                    current_section = {
                        "section_name": "Unknown Section",
                        "page_number": page_num,
                        "text": "",
                        "subsections": []
                    }
                    pdf_data["sections"].append(current_section)
                current_section["subsections"].append(current_subsection)

            else:
                if current_subsection:
                    current_subsection["text"] += line + "\n"
                elif current_section:
                    current_section["text"] += line + "\n"

    for sec in pdf_data["sections"]:
        sec["text"] = sec["text"].strip()
        for sub in sec["subsections"]:
            sub["text"] = sub["text"].strip()

    return pdf_data


def extract_multiple_pdfs(pdf_folder, output_json="all_pdfs_structure.json"):
    results = []

    for file_name in os.listdir(pdf_folder):
        if file_name.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, file_name)
            print(f"Processing: {pdf_path}")
            pdf_data = extract_pdf_structure(pdf_path)
            results.append(pdf_data)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"PDF structures saved to {output_json}")
    return results


# ---------------- CHUNKING & FAISS ---------------- #

model = SentenceTransformer("all-MiniLM-L6-v2")

def flatten_pdf_data(all_pdf_data):
    chunks = []
    for pdf in all_pdf_data:
        pdf_name = pdf["pdf_name"]

        for section in pdf["sections"]:
            if section["text"]:
                chunks.append({
                    "pdf_name": pdf_name,
                    "location": section["section_name"],
                    "page": section["page_number"],
                    "text": section["text"]
                })

            for sub in section["subsections"]:
                if sub["text"]:
                    chunks.append({
                        "pdf_name": pdf_name,
                        "location": f"{section['section_name']} -> {sub['subsection_name']}",
                        "page": sub["page_number"],
                        "text": sub["text"]
                    })
    return chunks


def build_faiss_index(chunks, faiss_index_path="pdf_index.faiss", metadata_path="pdf_metadata.json"):
    texts = [c["text"] for c in chunks]

    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)

    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    faiss.write_index(index, faiss_index_path)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"FAISS index saved to {faiss_index_path}")
    print(f"Metadata saved to {metadata_path}")

    return index


def search_faiss(query, index, metadata_path="pdf_metadata.json", top_k=6):
    with open(metadata_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)

    D, I = index.search(q_emb, top_k)

    results = []
    for idx, score in zip(I[0], D[0]):
        if idx == -1:
            continue
        result = chunks[idx].copy()
        result["score"] = float(score)
        results.append(result)

    return results


# ---------------- LLM SUMMARIZATION ---------------- #

# Load local LLM (adjust path in config.json)
with open("config.json") as f:
    cfg = json.load(f)

llm = Llama(model_path=cfg["local_llm_path"], n_ctx=2048, n_threads=6)  # use more threads if available

def generate_summary_llm(results, query):
    combined_text = "\n\n".join(
        [f"From {r['pdf_name']} (Page {r['page']}, Section: {r['location']}): {r['text']}" for r in results]
    )
    prompt = f"""
    You are a helpful assistant. Summarize the following retrieved passages in relation to the query: "{query}".
    Only include the most important findings. Avoid redundancy.

    Passages:
    {combined_text}
    """

    output = llm(prompt, max_tokens=256)

    # ✅ Safely extract the summary depending on version
    if "choices" in output and len(output["choices"]) > 0:
        return output["choices"][0].get("text", "").strip()
    elif "text" in output:
        return output["text"].strip()
    else:
        return str(output)



# ---------------- MAIN PIPELINE ---------------- #

if __name__ == "__main__":
    pdf_folder = "C:/Users/User/Desktop/traceability/data"  # change this to your folder path

    # Step 1: Extract structure from multiple PDFs
    all_pdf_data = extract_multiple_pdfs(pdf_folder)

    # Step 2: Flatten into chunks
    chunks = flatten_pdf_data(all_pdf_data)

    # Step 3: Build FAISS index
    index = build_faiss_index(chunks)

    # Step 4: Example query
    query = "What is technical report writing?"
    results = search_faiss(query, index)

    print("\nTop results:")
    for r in results:
        print(f"[{r['pdf_name']} | {r['location']} | Page {r['page']}] (score={r['score']:.4f})")
    top_results=results[:3]
    print("\n--- Summary ---")
    summary = generate_summary_llm(top_results, query)
    print(summary)
