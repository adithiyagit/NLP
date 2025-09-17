import spacy
import pdfplumber
import json

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# === Step 1: Extract text from PDF ===
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        all_text = ""
        for page in pdf.pages:
            all_text += page.extract_text() + "\n"
    return all_text

# === Step 2: Load Resume PDF ===
pdf_path = "Data scientist- Adithiya Vinu.pdf"  # Make sure this file is in the same directory
resume_text = extract_text_from_pdf(pdf_path)

# === Step 3: Process with spaCy ===
doc = nlp(resume_text)

# === Step 4: Extract Named Entities ===
entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

# === Step 5: Save as JSON ===
json_output = json.dumps(entities, indent=4, ensure_ascii=False)

with open("resume_entities.json", "w", encoding="utf-8") as f:
    f.write(json_output)

# === Step 6: Print Summary ===
print("🔍 Extracted Entities:")
print(json_output)
print("✅ JSON saved as 'resume_entities.json'")
