import os
import shutil
from openai import OpenAI, OpenAIError
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pdf2docx import Converter
from docx import Document
from docx2pdf import convert
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise RuntimeError("Missing GITHUB_TOKEN in environment")

# Initialize GitHub AI client
client = OpenAI(
    base_url="https://models.github.ai/inference",
    api_key=GITHUB_TOKEN,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")

# FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory for uploads
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def tailor_bullet(section_name: str, job_description: str, original_text: str) -> str:
    """Call GitHub GPT-4o to rewrite a single bullet point or skill."""
    prompt = f"Rewrite this {section_name} bullet point or skill to match the job description, returning a single bullet point. Job Description: {job_description}\nOriginal Text: {original_text}"
    messages = [
        {"role": "system", "content": "You are an expert resume writer."},
        {"role": "user", "content": prompt}
    ]
    try:
        response = client.chat.completions.create(
            model="openai/gpt-4.1-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=150,
            top_p=1.0
        )
        return response.choices[0].message.content.strip()
    except OpenAIError as e:
        logger.error(f"GitHub AI error: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"GitHub AI API error: {e}")

def is_bullet_point(paragraph) -> bool:
    """Check if a paragraph is a bullet point based on style or text prefix."""
    text = paragraph.text.strip()
    return paragraph.style.name.startswith("List") or text.startswith(("-", "•", "*"))

def process_docx(docx_path: str, job_description: str):
    """Process the DOCX file by tailoring each bullet point or skill individually."""
    doc = Document(docx_path)
    current_section = None

    for para in doc.paragraphs:
        text = para.text.strip().lower()
        # Identify section headers
        if text in ["objective", "skills", "experience"]:
            current_section = text.capitalize()
            continue
        # Process bullet points or skills under the current section
        if current_section and (is_bullet_point(para) or current_section == "Skills") and para.text.strip():
            original_text = para.text.strip()
            tailored_text = tailor_bullet(current_section, job_description, original_text)
            # Replace text in-place, preserving style
            para.clear()
            para.add_run(tailored_text)

    tailored_docx_path = docx_path.replace(".docx", "_tailored.docx")
    doc.save(tailored_docx_path)
    return tailored_docx_path

def process_resume(resume_pdf: str, job_description: str) -> tuple[str, str]:
    """Process the resume by converting to DOCX, tailoring bullet points, and converting back to PDF."""
    try:
        # Convert PDF to DOCX
        temp_docx = resume_pdf.replace(".pdf", ".docx")
        cv = Converter(resume_pdf)
        cv.convert(temp_docx)
        cv.close()

        # Process the DOCX by tailoring bullet points and skills
        tailored_docx = process_docx(temp_docx, job_description)

        # Convert tailored DOCX to PDF
        tailored_pdf = tailored_docx.replace(".docx", ".pdf")
        convert(tailored_docx, tailored_pdf)

        # Clean up temporary files
        os.remove(resume_pdf)
        os.remove(temp_docx)

        return tailored_pdf, tailored_docx
    except Exception as e:
        logger.error(f"Error processing resume: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing resume")

@app.post("/tailor-resume")
async def tailor_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):
    """Endpoint to upload a resume PDF and job description, returning tailored files."""
    pdf_path = os.path.join(UPLOAD_DIR, resume.filename)
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(resume.file, f)

    tailored_pdf, tailored_docx = process_resume(pdf_path, job_description)
    return {"pdf_path": tailored_pdf, "docx_path": tailored_docx}

@app.get("/")
async def health_check():
    return {"status": "OK"}
