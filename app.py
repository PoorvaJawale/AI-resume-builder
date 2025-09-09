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
from openai import OpenAI  # ✅ use new SDK
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')  # optional but recommended for lemmatization

from nltk.stem import WordNetLemmatizer

# Load environment variables
load_dotenv()

app = Flask(__name__)
GENERATED_DIR = Path("generated")
GENERATED_DIR.mkdir(exist_ok=True)

# -------------------------
# OpenAI client setup
# -------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
lemmatizer = WordNetLemmatizer()

if not OPENAI_API_KEY:
    print("⚠️ No OpenAI API key found. Using fallback rule-based logic.")
    client = None
else:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # -------------------------
    # Test OpenAI API key
    # -------------------------
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello, test if API is working!"}]
        )
        print("✅ OpenAI API key is working! Sample response:")
        print(response.choices[0].message.content)
    except Exception as e:
        print("❌ OpenAI test failed:", e)
        print("Using fallback rule-based logic instead.")
        client = None



# -------------------------
# Helpers
# -------------------------

def extract_text_from_pdf(file_path):
    """Extract text from PDF using pdfplumber."""
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
    """Extract text from DOCX file."""
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
    """Lowercase, remove non-alphanumeric, lemmatize, remove stopwords, remove 1-letter words."""
    tokens = re.findall(r'\b\w+\b', (text or "").lower())
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens

def compute_match_score(resume_text, job_desc):
    """Compute keyword match score between resume and job description using lemmatized tokens."""
    job_tokens = set(tokenize(job_desc))
    resume_tokens = set(tokenize(resume_text))
    common = job_tokens.intersection(resume_tokens)

    if not job_tokens:
        score = 0
    else:
        score = (len(common) / len(job_tokens)) * 100

    # Debug prints (remove in production)
    print("Job tokens:", job_tokens)
    print("Resume tokens:", resume_tokens)
    print("Common tokens:", common)

    return round(score, 1), sorted(list(common)), sorted(list(job_tokens - common))


def call_openai_optimize(resume_text, job_desc):
    """Call OpenAI API to optimize resume."""
    if not client:
        # Fallback: simple rule-based improvements
        improved = resume_text.replace("responsible for", "implemented").replace("worked on", "implemented")
        return {
            "optimized_resume": improved,
            "suggested_improvements": ["Replace weak verbs with active verbs", "Add measurable outcomes (percent, numbers)"],
            "roadmap": [
                {"step": "Learn missing skill X", "time_estimate": "2 weeks"},
                {"step": "Build a project", "time_estimate": "2-3 weeks"},
                {"step": "Practice interviews & update resume", "time_estimate": "1 week"}
            ]
        }

    system_prompt = (
        "You are an expert career coach and resume writer. "
        "You will receive a candidate resume (raw text) and a target job description. "
        "Produce a JSON object ONLY with keys: optimized_resume (string), suggested_improvements (array of short strings), "
        "roadmap (array of objects with keys step and time_estimate). Keep outputs concise."
    )

    user_prompt = f"Resume:\n{resume_text}\n\nTarget job description:\n{job_desc}\n\nReturn JSON ONLY."

    try:
        resp = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=800
        )

        content = resp.choices[0].message.content.strip()

        # Try parse JSON
        try:
            data = json.loads(content)
        except Exception:
            idx = content.find('{')
            if idx != -1:
                data = json.loads(content[idx:])
            else:
                raise
        return data
    except Exception as e:
        print("⚠️ OpenAI failed or quota exceeded:", e)
        # fallback logic
        improved = resume_text.replace("responsible for", "implemented").replace("worked on", "implemented")
        return {
            "optimized_resume": improved,
            "suggested_improvements": ["(Fallback) Manually review verbs & metrics."],
            "roadmap": [
                {"step": "Learn missing skill X", "time_estimate": "2 weeks"},
                {"step": "Build a project", "time_estimate": "2-3 weeks"},
                {"step": "Practice interviews & update resume", "time_estimate": "1 week"}
            ]
        }



def save_docx_from_text(text, filename_path):
    """Save plain text into a DOCX file."""
    doc = Document()
    for line in text.splitlines():
        doc.add_paragraph(line)
    doc.save(filename_path)


# -------------------------
# Routes
# -------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    job_desc = request.form.get("job_desc", "").strip()
    job_desc = re.sub(r'\s+', ' ', job_desc)  # remove extra spaces/newlines

    uploaded = request.files.get("resume_file")

    if not uploaded or uploaded.filename == "":
        flash("Please upload a resume PDF or DOCX file.")
        return redirect(url_for("index"))

    # Save temp file
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=os.path.splitext(uploaded.filename)[1] or ".pdf")
    uploaded.save(tmp_path)

    # Extract text
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

    # Compute scores
    before_score, matched_keywords, missing_keywords = compute_match_score(resume_text, job_desc)

    # AI optimization
    ai_result = call_openai_optimize(resume_text, job_desc)
    optimized_resume = ai_result.get("optimized_resume", resume_text)
    suggested_improvements = ai_result.get("suggested_improvements", [])
    roadmap = ai_result.get("roadmap", [])

    after_score, matched_after, missing_after = compute_match_score(optimized_resume, job_desc)

    # Save optimized resume
    file_id = uuid.uuid4().hex
    out_path = GENERATED_DIR / f"{file_id}.docx"
    save_docx_from_text(optimized_resume, out_path)

    # Remove temp file
    try:
        os.close(tmp_fd)
        os.remove(tmp_path)
    except Exception:
        pass

    return render_template("results.html",
                           before_score=before_score,
                           after_score=after_score,
                           matched_keywords=matched_keywords,
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
