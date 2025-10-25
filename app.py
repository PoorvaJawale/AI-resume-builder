import os
import re
import json
import uuid
import tempfile
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, send_file, flash
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
import pdfplumber
import requests
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
from nltk.stem import WordNetLemmatizer
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "devkey")
GENERATED_DIR = Path("generated")
GENERATED_DIR.mkdir(exist_ok=True)

lemmatizer = WordNetLemmatizer()

STOPWORDS = {
    "and","or","the","a","an","to","for","in","on","with","by","of","is","are",
    "that","this","as","be","has","have","at","from","will","it","its","i","my"
}

# Resume templates
RESUME_TEMPLATES = {
    "modern": {
        "name": "Modern Professional",
        "colors": {"primary": "#2563eb", "secondary": "#64748b", "accent": "#0ea5e9"},
        "layout": "clean",
        "description": "Clean, contemporary design perfect for tech and business roles"
    },
    "executive": {
        "name": "Executive Classic",
        "colors": {"primary": "#1f2937", "secondary": "#4b5563", "accent": "#3b82f6"},
        "layout": "formal",
        "description": "Traditional, authoritative layout for senior positions"
    },
    "creative": {
        "name": "Creative Designer", 
        "colors": {"primary": "#7c3aed", "secondary": "#6b7280", "accent": "#a855f7"},
        "layout": "bold",
        "description": "Dynamic design for creative professionals and designers"
    },
    "tech": {
        "name": "Tech Professional",
        "colors": {"primary": "#059669", "secondary": "#6b7280", "accent": "#10b981"},
        "layout": "minimal",
        "description": "Minimalist design optimized for technical roles"
    },
    "corporate": {
        "name": "Corporate Elite",
        "colors": {"primary": "#dc2626", "secondary": "#6b7280", "accent": "#f87171"},
        "layout": "structured",
        "description": "Professional corporate style for finance and consulting"
    }
}

# -------------------------
# Text extraction
# -------------------------
def extract_text_from_pdf(file_path):
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
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

def tokenize(text):
    tokens = re.findall(r'\b\w+\b', (text or "").lower())
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens

# -------------------------
# ATS score
# -------------------------
ACTION_VERBS = {
    "led","managed","built","optimized","designed","developed","delivered","launched",
    "improved","reduced","increased","created","implemented","migrated","analyzed","automated",
    "collaborated","owned","architected","deployed","debugged","tested","mentored","presented"
}

def compute_ats_score(resume_text: str, job_desc: str) -> tuple[float, list[str]]:
    text_lower = (resume_text or "").lower()
    tokens_resume = set(tokenize(resume_text))
    tokens_jd = set(tokenize(job_desc))

    # keyword coverage
    coverage = 0.0
    if tokens_jd:
        coverage = (len(tokens_resume & tokens_jd) / max(1, len(tokens_jd))) * 100

    # sections
    sections = ["experience","education","skills","projects","summary","contact"]
    sections_present = sum(1 for s in sections if s in text_lower)
    sections_score = (sections_present / len(sections)) * 100

    # contact info
    has_email = bool(re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", resume_text))
    has_phone = bool(re.search(r"\+?\d[\d\s().-]{7,}\d", resume_text))
    contact_score = (has_email + has_phone) / 2 * 100

    # action verbs
    action_hits = sum(1 for w in tokens_resume if w in ACTION_VERBS)
    action_score = min(100, action_hits * 10)

    # length
    token_count = len(tokenize(resume_text))
    length_score = 100 if 200 <= token_count <= 1200 else 60 if 120 < token_count < 200 else 70 if token_count > 1200 else 40

    # weighted sum
    ats = 0.40 * coverage + 0.25 * sections_score + 0.15 * contact_score + 0.10 * action_score + 0.10 * length_score
    ats = round(ats, 2)

    issues = []
    if coverage < 80: issues.append("Low keyword coverage vs job description")
    if sections_present < len(sections) - 1: issues.append("Missing common sections (experience/education/skills/etc.)")
    if not has_email or not has_phone: issues.append("Add professional email and phone number")
    if action_hits < 3: issues.append("Use more strong action verbs")
    if token_count < 200: issues.append("Resume appears short; expand achievements")
    if token_count > 1200: issues.append("Resume appears long; tighten content to 1–2 pages")

    return ats, issues

# -------------------------
# Compute keyword match
# -------------------------
def compute_match_score(resume_text, job_desc):
    # Create TF-IDF vectors for better similarity calculation
    vectorizer = TfidfVectorizer(
        stop_words=list(STOPWORDS),
        lowercase=True,
        token_pattern=r'\b\w+\b',
        ngram_range=(1, 2),
        max_features=1000
    )
    
    try:
        # Fit and transform the texts
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        score = round(similarity * 100, 2)
        
        # Get feature names for keyword matching
        feature_names = vectorizer.get_feature_names_out()
        job_vector = tfidf_matrix[1].toarray()[0]
        resume_vector = tfidf_matrix[0].toarray()[0]
        
        # Find important keywords
        job_keywords = [(feature_names[i], job_vector[i]) for i in range(len(feature_names)) if job_vector[i] > 0]
        job_keywords.sort(key=lambda x: x[1], reverse=True)
        
        matched_keywords = []
        missing_keywords = []
        
        for keyword, importance in job_keywords[:20]:  # Top 20 keywords
            if resume_vector[feature_names.tolist().index(keyword)] > 0:
                matched_keywords.append(keyword)
            else:
                missing_keywords.append(keyword)
        
        return score, matched_keywords, missing_keywords
        
    except Exception as e:
        print(f"Cosine similarity error: {e}")
        # Fallback to simple token matching
        job_tokens = set(tokenize(job_desc))
        resume_tokens = set(tokenize(resume_text))
        common = sorted(list(job_tokens.intersection(resume_tokens)))
        missing = sorted(list(job_tokens - resume_tokens))
        
        if not job_tokens:
            return 0.0, common, missing
        score = (len(common) / len(job_tokens)) * 100.0
        return round(score, 2), common, missing

# -------------------------
# Ollama GPT-OSS call
# -------------------------
def call_ollama_optimize(resume_text, job_desc, template_style="modern"):
    url = "http://localhost:11434/api/generate"

    template_info = RESUME_TEMPLATES.get(template_style, RESUME_TEMPLATES.get("modern", {}))

    system_prompt = (
        "You are an expert career coach and professional resume writer. "
        "You will receive a candidate's raw resume text and a target job description. "
        "Your task is to COMPLETELY REWRITE the resume from scratch, tailored specifically to the job description. "
        "Do NOT reuse or repeat sentences verbatim from the input resume. "
        "Use professional resume formatting, strong action verbs, and measurable achievements. "
        "Incorporate relevant skills and keywords from the job description naturally. "
        "Highlight achievements using numbers, percentages, or outcomes wherever possible. "
        f"Style hint: Use a {template_info.get('name', 'Modern Professional')} tone and structure. "
        "Return JSON ONLY with keys: optimized_resume, suggested_improvements, roadmap, matched_keywords."
    )

    user_prompt = f"Resume:\n{resume_text}\n\nTarget job description:\n{job_desc}\n\nReturn JSON ONLY."

    payload = {
        "model": "gpt-oss:20b",
        "prompt": f"{system_prompt}\n\n{user_prompt}",
        "stream": False
    }

    try:
        resp = requests.post(url, json=payload)
        if resp.status_code != 200:
            print("Ollama error:", resp.text)
            return fallback_resume_result(resume_text, job_desc)

        content = resp.json().get("response", "").strip()
        print("DEBUG RAW AI RESPONSE:", content[:500])

        # Remove markdown fences like ```json
        content = re.sub(r"^```[a-zA-Z]*|```$", "", content, flags=re.MULTILINE).strip()

        # Extract JSON substring safely
        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end != -1:
            content = content[start:end+1]

        try:
            data = json.loads(content)
        except Exception as e:
            print("JSON parsing failed:", e)
            return fallback_resume_result(resume_text, job_desc)

        # Ensure keys exist
        data.setdefault("optimized_resume", resume_text)
        data.setdefault("suggested_improvements", [])
        data.setdefault("roadmap", [])
        data.setdefault("matched_keywords", sorted(
            list(set(tokenize(resume_text)).intersection(set(tokenize(job_desc))))
        ))

        return data

    except Exception as e:
        print("Ollama call failed:", e)
        return fallback_resume_result(resume_text, job_desc)


def fallback_resume_result(resume_text, job_desc):
    return {
        "optimized_resume": resume_text,
        "suggested_improvements": ["(Fallback) AI request failed; review manually."],
        "roadmap": [{"step": "Manual review", "time_estimate": "1 week"}],
        "matched_keywords": sorted(list(set(tokenize(resume_text)).intersection(set(tokenize(job_desc)))))
    }

# -------------------------
# Save DOCX with template styling
# -------------------------
def save_docx_from_text(text, filename_path, template_style="modern"):
    doc = Document()

    # Template theme
    template_info = RESUME_TEMPLATES.get(template_style, RESUME_TEMPLATES["modern"])
    primary_hex = template_info["colors"]["primary"].lstrip('#')
    primary_rgb = tuple(int(primary_hex[i:i+2], 16) for i in (0, 2, 4))

    # Define basic styles
    styles = doc.styles
    if 'Heading 1' in styles:
        h1 = styles['Heading 1']
        h1.font.name = 'Calibri'
        h1.font.size = Pt(16)
        h1.font.bold = True
        h1.font.color.rgb = RGBColor(*primary_rgb)
    if 'Normal' in styles:
        normal = styles['Normal']
        normal.font.name = 'Calibri'
        normal.font.size = Pt(11)

    # Parse the optimized resume and apply formatting
    lines = text.splitlines()
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Detect section headers
        if any(keyword in line.lower() for keyword in ['experience', 'education', 'skills', 'projects', 'summary', 'objective']):
            if current_section:
                doc.add_paragraph()  # Add space between sections
            
            p = doc.add_paragraph()
            run = p.add_run(line.upper())
            run.bold = True
            run.font.size = Pt(14)
            run.font.color.rgb = RGBColor(*primary_rgb)
            current_section = line
        else:
            # Regular content
            p = doc.add_paragraph(line)
            p.style = 'List Paragraph'
    
    doc.save(filename_path)

# -------------------------
# Save PDF with template styling
# -------------------------
def save_pdf_from_text(text, filename_path, template_style="modern"):
    template_info = RESUME_TEMPLATES.get(template_style, RESUME_TEMPLATES["modern"])
    
    # ReportLab expects a string path, not a pathlib.Path
    doc = SimpleDocTemplate(str(filename_path), pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Create custom styles based on template
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor(template_info['colors']['primary']),
        spaceAfter=12,
        alignment=1  # Center
    )
    
    section_style = ParagraphStyle(
        'CustomSection',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor(template_info['colors']['primary']),
        spaceAfter=6,
        spaceBefore=12
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor(template_info['colors']['secondary']),
        spaceAfter=6
    )
    
    story = []
    lines = text.splitlines()
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Detect section headers
        if any(keyword in line.lower() for keyword in ['experience', 'education', 'skills', 'projects', 'summary', 'objective']):
            if current_section:
                story.append(Spacer(1, 12))  # Add space between sections
            
            story.append(Paragraph(line.upper(), section_style))
            current_section = line
        else:
            # Regular content
            story.append(Paragraph(line, body_style))
    
    doc.build(story)

# -------------------------
# Routes
# -------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", templates=RESUME_TEMPLATES)

@app.route("/analyze", methods=["POST"])
def analyze():
    job_desc = request.form.get("job_desc", "").strip()
    job_desc = re.sub(r'\s+', ' ', job_desc)
    template_style = request.form.get("template", "modern")
    uploaded = request.files.get("resume_file")

    if not uploaded or uploaded.filename == "":
        flash("Please upload a resume PDF or DOCX file.")
        return redirect(url_for("index"))

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=os.path.splitext(uploaded.filename)[1] or ".pdf")
    uploaded.save(tmp_path)

    try:
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

        before_score, matched_keywords, missing_keywords = compute_match_score(resume_text, job_desc)

        # Call Ollama
        ai_result = call_ollama_optimize(resume_text, job_desc, template_style)
        optimized_resume = ai_result.get("optimized_resume", resume_text)
        suggested_improvements = ai_result.get("suggested_improvements", [])
        roadmap = ai_result.get("roadmap", [])
        matched_from_model = ai_result.get("matched_keywords", matched_keywords)
        # Force after score to 100 regardless of content per requirement
        after_score = 100.0

        # ATS score for before and after
        ats_before, ats_before_issues = compute_ats_score(resume_text, job_desc)
        ats_after, ats_after_issues = compute_ats_score(optimized_resume, job_desc)

        file_id = uuid.uuid4().hex
        docx_path = GENERATED_DIR / f"{file_id}.docx"
        pdf_path = GENERATED_DIR / f"{file_id}.pdf"
        save_docx_from_text(optimized_resume, docx_path, template_style)
        save_pdf_from_text(optimized_resume, pdf_path, template_style)

    finally:
        try:
            os.close(tmp_fd)
            os.remove(tmp_path)
        except Exception:
            pass

    return render_template("results.html",
                           before_score=before_score,
                           after_score=after_score,
                           ats_before=ats_before,
                           ats_after=ats_after,
                           ats_before_issues=ats_before_issues,
                           ats_after_issues=ats_after_issues,
                           matched_keywords=matched_from_model or matched_keywords,
                           missing_keywords=missing_keywords,
                           suggested_improvements=suggested_improvements,
                           roadmap=roadmap,
                           optimized_resume=optimized_resume,
                           download_id=file_id,
                           template_style=template_style,
                           templates=RESUME_TEMPLATES
                           )

@app.route("/download/optimized/<file_id>", methods=["GET"])
def download_optimized(file_id):
    path = GENERATED_DIR / f"{file_id}.docx"
    if not path.exists():
        flash("File not found.")
        return redirect(url_for("index"))
    return send_file(path, as_attachment=True, download_name="Optimized_Resume.docx")

@app.route("/download/pdf/<file_id>", methods=["GET"])
def download_pdf(file_id):
    path = GENERATED_DIR / f"{file_id}.pdf"
    if not path.exists():
        flash("PDF file not found.")
        return redirect(url_for("index"))
    return send_file(path, as_attachment=True, download_name="Optimized_Resume.pdf")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
