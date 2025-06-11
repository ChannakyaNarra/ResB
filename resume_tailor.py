from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from openai import OpenAI, OpenAIError
import shutil
from pdf2docx import Converter
from docx import Document
from docx2pdf import convert
import logging

from dotenv import load_dotenv
import os

app = FastAPI()

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise RuntimeError("Missing GITHUB_TOKEN in environment")

# Initialize the GitHub AI client
client = OpenAI(
    base_url="https://models.github.ai/inference",
    api_key=GITHUB_TOKEN,
)

logger = logging.getLogger("uvicorn.error")

def tailor_section(job_description: str, original_text: str) -> str:
    # Build the chat messages
    messages = [
        {"role": "system", "content": "You are an expert resume writer."},
        {"role": "user", "content": (
            f"Rewrite this resume section to match the job description:\n\n"
            f"Job Description:\n{job_description}\n\n"
            f"Original Section:\n{original_text}\n\n"
            "Tailored Section:"
        )}
    ]

    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=500,
            top_p=1.0
        )
        return response.choices[0].message.content.strip()


    except OpenAIError as e:
        logger.error(f"GitHub AI error: {e}", exc_info=True)
        raise HTTPException(
            status_code=502,
            detail=f"GitHub AI API error: {e}"
        )

def process_resume(resume_path: str, job_description: str) -> tuple:
    # Convert PDF to DOCX
    docx_path = resume_path.replace(".pdf", ".docx")
    cv = Converter(resume_path)
    cv.convert(docx_path)
    cv.close()

    # Load DOCX
    doc = Document(docx_path)

    # Identify sections (simplified heuristic)
    sections = {"Summary": [], "Skills": [], "Experience": []}
    current_section = None
    for para in doc.paragraphs:
        text = para.text.strip().lower()
        if "summary" in text and not current_section:
            current_section = "Summary"
        elif "skills" in text and not current_section:
            current_section = "Skills"
        elif "experience" in text and not current_section:
            current_section = "Experience"
        elif current_section and text and para.text not in sections:
            sections[current_section].append(para)

    # Tailor each section
    for section, paragraphs in sections.items():
        if paragraphs:
            original_text = "\n".join([p.text for p in paragraphs if p.text.strip()])
            if original_text:
                tailored_text = tailor_section(job_description, original_text)
                for para in paragraphs:
                    para.text = ""  # Clear original
                paragraphs[0].text = tailored_text  # Replace with tailored

    # Save tailored DOCX
    tailored_docx = os.path.join(UPLOAD_DIR, f"tailored_{os.path.basename(resume_path).replace('.pdf', '.docx')}")
    doc.save(tailored_docx)

    # Convert to PDF
    tailored_pdf = tailored_docx.replace(".docx", ".pdf")
    convert(tailored_docx, tailored_pdf)

    # Clean up temporary files
    os.remove(resume_path)
    os.remove(docx_path)

    return tailored_pdf, tailored_docx


@app.post("/tailor-resume")
async def tailor_resume(resume: UploadFile = File(...), job_description: str = Form(...)):
    # Save uploaded resume
    resume_path = os.path.join(UPLOAD_DIR, resume.filename)
    with open(resume_path, "wb") as f:
        shutil.copyfileobj(resume.file, f)

    # Process the resume
    tailored_pdf, tailored_docx = process_resume(resume_path, job_description)

    # Return file paths (in production, use unique IDs and serve files)
    return {
        "message": "Resume tailored successfully",
        "pdf_path": tailored_pdf,
        "docx_path": tailored_docx
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
