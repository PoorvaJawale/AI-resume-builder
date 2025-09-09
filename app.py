# app.py (drop-in replacement)
import os
import re
import json
import uuid
import tempfile
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, send_file, flash
from docx import Document
import pdfplumber
from dotenv import load_dotenv

# -------------------------
# New imports
# -------------------------
import numpy as np  # for cosine similarity of embeddings
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
from nltk.stem import WordNetLemmatizer

# OpenAI import (guarded)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "devkey")
GENERATED_DIR = Path("generated")
GENERATED_DIR.mkdir(exist_ok=True)

# -------------------------
# OpenAI client setup
# -------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = None
if OPENAI_API_KEY and OpenAI is not None:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        # quick smoke test (non-fatal)
        try:
            _ = client.embeddings.create(model="text-embedding-3-small", input=["test"])
            print("OpenAI embeddings available")
        except Exception as e:
            print("OpenAI test error (embeddings):", e)
            # keep client as-is; we'll handle exceptions when calling
    except Exception as e:
        print("OpenAI initialization failed:", e)
        client = None
else:
    print("No OpenAI key/client available - running in fallback mode.")

# -------------------------
# Helpers
# -------------------------
lemmatizer = WordNetLemmatizer()

def extract_text_from_pdf(file_path):
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
    except Exception as e:
        print("pdfplumber error:", e)
    return text.strip()

def extract_text_from_docx(file_path):
    text = ""
    try:
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print("docx extraction error:", e)
    return text.strip()

STOPWORDS = {
    "and","or","the","a","an","to","for","in","on","with","by","of","is","are",
    "that","this","as","be","has","have","at","from","will","it","its","i","my"
}

def tokenize(text):
    tokens = re.findall(r'\b\w+\b', (text or "").lower())
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens

# -------------------------
# New: embedding helpers (uses OpenAI embeddings)
# -------------------------
def get_embedding(text, model_name="text-embedding-3-small"):
    """
    Return embedding vector using OpenAI embeddings (if client exists).
    Raises Exception if client is None or API fails.
    """
    if not client:
        raise RuntimeError("OpenAI client not available")
    # Normalize
    text = (text or "").replace("\n", " ")
    resp = client.embeddings.create(model=model_name, input=[text])
    emb = resp.data[0].embedding
    return np.array(emb, dtype=float)

def cosine_similarity(a: np.ndarray, b: np.ndarray):
    if a is None or b is None:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

# -------------------------
# compute_match_score changes
# -------------------------
def compute_match_score(resume_text, job_desc):
    """
    Compute match score and matched/missing keywords.
    - If OpenAI client is available, compute semantic similarity using embeddings for a numeric score.
    - Also compute token overlap (lemmatized) to supply matched_keywords and missing_keywords lists.
    Returns: (score_percent(float), matched_keywords(list), missing_keywords(list))
    ### CHANGE: switched to embeddings-based numeric score when possible
    """
    # token overlap for matched/missing lists (keeps previous behavior)
    job_tokens = set(tokenize(job_desc))
    resume_tokens = set(tokenize(resume_text))
    common = sorted(list(job_tokens.intersection(resume_tokens)))
    missing = sorted(list(job_tokens - resume_tokens))

    # Try semantic score using embeddings when available
    if client:
        try:
            job_emb = get_embedding(job_desc)
            resume_emb = get_embedding(resume_text)
            sim = cosine_similarity(resume_emb, job_emb)
            score = round(sim * 100.0, 2)
            return score, common, missing
        except Exception as e:
            print("Embedding-based score failed:", e)
            # fall through to keyword score

    # Fallback: keyword ratio score (previous logic)
    if not job_tokens:
        return 0.0, common, missing
    score = (len(common) / len(job_tokens)) * 100.0
    return round(score, 2), common, missing

# -------------------------
# call_openai_optimize changes
# -------------------------
def call_openai_optimize(resume_text, job_desc):
    """
    Ask OpenAI chat model to return an optimized resume and metadata as JSON.
    Always returns a dictionary with keys:
        - optimized_resume
        - suggested_improvements
        - roadmap
        - matched_keywords
    """
    # If OpenAI client is not available, return the original resume but keep keys
    if not client:
        return {
            "optimized_resume": resume_text,  # keeps template happy
            "suggested_improvements": [],
            "roadmap": [],
            "matched_keywords": sorted(list(set(tokenize(resume_text)).intersection(set(tokenize(job_desc)))))
        }

    # System prompt for AI rewrite
    system_prompt = (
    "You are an expert career coach and professional resume writer. "
    "You will receive a candidate's raw resume text and a target job description. "
    "Your task is to COMPLETELY REWRITE the resume from scratch, tailored specifically to the job description. "
    "Do NOT reuse or repeat sentences verbatim from the input resume. "
    "Instead, use professional resume formatting, strong action verbs, and measurable achievements. "
    "Incorporate relevant skills and keywords from the job description naturally. "
    "Highlight achievements using numbers, percentages, or outcomes wherever possible. "
    "Always return your output as valid JSON ONLY with the following keys:\n"
    "1. optimized_resume (string) → the fully rewritten professional resume.\n"
    "2. suggested_improvements (array of short strings) → list of resume writing suggestions.\n"
    "3. roadmap (array of objects with keys: step, time_estimate) → career improvement plan.\n"
    "4. matched_keywords (array of strings) → keywords from the job description found in the rewritten resume.\n\n"
    "Return JSON only. Do not include explanations or commentary outside of JSON."
)


    user_prompt = f"Resume:\n{resume_text}\n\nTarget job description:\n{job_desc}\n\nReturn JSON ONLY."

    try:
        resp = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
            max_tokens=1500
        )

        content = resp.choices[0].message.content.strip()

        # Parse JSON robustly
        try:
            data = json.loads(content)
        except Exception:
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1:
                data = json.loads(content[start:end+1])
            else:
                data = {}

        # Ensure all required keys exist
        data.setdefault("optimized_resume", resume_text)
        data.setdefault("suggested_improvements", [])
        data.setdefault("roadmap", [])
        if "matched_keywords" not in data:
            data["matched_keywords"] = sorted(list(set(tokenize(resume_text)).intersection(set(tokenize(job_desc)))))

        return data

    except Exception as e:
        print("OpenAI chat failed:", e)
        # fallback: just return the original text with keys (no weak "implemented" replacements)
        return {
            "optimized_resume": resume_text,
            "suggested_improvements": ["(Fallback) AI optimization failed; review manually."],
            "roadmap": [{"step": "Manual review", "time_estimate": "1 week"}],
            "matched_keywords": sorted(list(set(tokenize(resume_text)).intersection(set(tokenize(job_desc)))))
        }


def save_docx_from_text(text, filename_path):
    doc = Document()
    for line in text.splitlines():
        doc.add_paragraph(line)
    doc.save(filename_path)

# -------------------------
# Routes (mostly unchanged)
# -------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    job_desc = request.form.get("job_desc", "").strip()
    job_desc = re.sub(r'\s+', ' ', job_desc)
    uploaded = request.files.get("resume_file")

    if not uploaded or uploaded.filename == "":
        flash("Please upload a resume PDF or DOCX file.")
        return redirect(url_for("index"))

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=os.path.splitext(uploaded.filename)[1] or ".pdf")
    uploaded.save(tmp_path)

    ext = uploaded.filename.lower()
    if ext.endswith(".pdf"):
        resume_text = extract_text_from_pdf(tmp_path)
    elif ext.endswith(".docx"):
        resume_text = extract_text_from_docx(tmp_path)
    else:
        try:
            with open(tmp_path, "r", encoding="utf-8") as f:
                resume_text = f.read()
        except Exception:
            resume_text = extract_text_from_pdf(tmp_path)

    if not resume_text:
        resume_text = "(No text could be extracted from the uploaded file.)"

    # Compute before score (embedding if possible)
    before_score, matched_keywords, missing_keywords = compute_match_score(resume_text, job_desc)

    # Ask OpenAI to optimize resume (or fallback)
    ai_result = call_openai_optimize(resume_text, job_desc)
    optimized_resume = ai_result.get("optimized_resume", resume_text)
    suggested_improvements = ai_result.get("suggested_improvements", [])
    roadmap = ai_result.get("roadmap", [])
    # prefer matched keywords from model if provided
    matched_from_model = ai_result.get("matched_keywords", matched_keywords)

    # Compute after score (embedding if possible)
    after_score, _, _ = compute_match_score(optimized_resume, job_desc)

    # Save optimized resume as .docx
    file_id = uuid.uuid4().hex
    out_path = GENERATED_DIR / f"{file_id}.docx"
    save_docx_from_text(optimized_resume, out_path)

    # cleanup
    try:
        os.close(tmp_fd)
        os.remove(tmp_path)
    except Exception:
        pass

    return render_template("results.html",
                           before_score=before_score,
                           after_score=after_score,
                           matched_keywords=matched_from_model or matched_keywords,
                           missing_keywords=missing_keywords,
                           suggested_improvements=suggested_improvements,
                           roadmap=roadmap,
                           optimized_resume=optimized_resume,
                           download_id=file_id
                           )

@app.route("/download/optimized/<file_id>", methods=["GET"])
def download_optimized(file_id):
    path = GENERATED_DIR / f"{file_id}.docx"
    if not path.exists():
        flash("File not found.")
        return redirect(url_for("index"))
    return send_file(path, as_attachment=True, download_name="Optimized_Resume.docx")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
