import os
import re
import json
import fitz  # PyMuPDF
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import streamlit as st

# ---------------- PDF STRUCTURE EXTRACTION ---------------- #
section_pattern = re.compile(r'^\d+\.\s+.+')
subsection_pattern = re.compile(r'^\d+\.\d+\s+.+')

def extract_pdf_structure(pdf_path):
    doc = fitz.open(pdf_path)
    pdf_data = {"pdf_name": os.path.basename(pdf_path), "sections": []}
    current_section, current_subsection = None, None

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue

            if section_pattern.match(line) and not subsection_pattern.match(line):
                current_section = {"section_name": line, "page_number": page_num, "text": "", "subsections": []}
                pdf_data["sections"].append(current_section)
                current_subsection = None

            elif subsection_pattern.match(line):
                current_subsection = {"subsection_name": line, "page_number": page_num, "text": ""}
                if current_section is None:
                    current_section = {"section_name": "Unknown Section", "page_number": page_num, "text": "", "subsections": []}
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

# ---------------- CHUNKING & FAISS ---------------- #
model = SentenceTransformer("all-MiniLM-L6-v2")

def flatten_pdf_data(all_pdf_data):
    chunks = []
    for pdf in all_pdf_data:
        pdf_name = pdf["pdf_name"]
        for section in pdf["sections"]:
            if section["text"]:
                chunks.append({"pdf_name": pdf_name, "location": section["section_name"], "page": section["page_number"], "text": section["text"]})
            for sub in section["subsections"]:
                if sub["text"]:
                    chunks.append({"pdf_name": pdf_name, "location": f"{section['section_name']} -> {sub['subsection_name']}", "page": sub["page_number"], "text": sub["text"]})
    return chunks

def build_faiss_index(chunks):
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index, chunks

def search_faiss(query, index, chunks, top_k=6):
    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    results = []
    for idx, score in zip(I[0], D[0]):
        if idx == -1: continue
        r = chunks[idx].copy()
        r["score"] = float(score)
        results.append(r)
    return results

# ---------------- STREAMLIT APP ---------------- #
st.title("📚 PDF Semantic Search with FAISS")
st.write("Upload multiple PDFs, build an index, and query them.")

uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
query = st.text_input("Enter your query")
search_btn = st.button("Search")

if uploaded_files and search_btn:
    all_pdf_data = []
    for uploaded_file in uploaded_files:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.read())
        all_pdf_data.append(extract_pdf_structure(uploaded_file.name))

    chunks = flatten_pdf_data(all_pdf_data)
    index, chunks = build_faiss_index(chunks)

    results = search_faiss(query, index, chunks)
    df = pd.DataFrame(results)[["pdf_name", "location", "page", "score", "text"]]
    st.write("### Search Results")
    st.dataframe(df)
