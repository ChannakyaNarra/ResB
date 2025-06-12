import os
import shutil
import uuid
import asyncio
import re
import logging
from openai import OpenAI, OpenAIError
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from docx import Document
from docx2pdf import convert
from dotenv import load_dotenv

# --- INITIALIZATION AND CONFIGURATION ---

# Load environment variables from .env file
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise RuntimeError("FATAL: Missing GITHUB_TOKEN in environment variables.")

# Initialize the AI client
try:
    client = OpenAI(
        base_url="https://models.github.ai/inference",
        api_key=GITHUB_TOKEN,
    )
except Exception as e:
    raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

# Set up logging to provide detailed insight into the process
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Resume Tailoring API v7 (Block-Aware)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory for storing files
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- CORE AI AND DOCUMENT PROCESSING LOGIC ---

API_CONCURRENCY_LIMIT = 4
semaphore = asyncio.Semaphore(API_CONCURRENCY_LIMIT)


async def tailor_content(section_name: str, job_description: str, original_text: str) -> str:
    """
    Asynchronously calls the AI model with a prompt tailored to the content type (bullet or paragraph).
    """
    async with semaphore:
        # Default prompt for standard bullet points
        prompt_instruction = (
            "Rewrite this single resume bullet point to be more impactful and align with the job description. "
            "**Crucially, the rewritten bullet point's word count MUST be within 10% (plus or minus) of the original's word count.** "
            "This is essential to preserve the document layout. "
            "Return only the single, rewritten bullet point text, without any prefixes like '•' or '-'."
        )

        # Specific prompt for paragraph-based sections like Objective or Summary
        if any(sec in section_name.upper() for sec in ["OBJECTIVE", "SUMMARY", "ABOUT"]):
            prompt_instruction = (
                "Rewrite this resume summary/objective paragraph to be more impactful and align with the job description. "
                "**The rewritten paragraph's word count MUST be almost identical to the original to preserve the document's layout.** "
                "Return only the rewritten paragraph text."
            )

        prompt = (
            f"{prompt_instruction}\n\n"
            f"Resume Section: '{section_name}'\n"
            f"Job Description: \"{job_description}\"\n\n"
            f"Original Text:\n---\n{original_text}\n---"
        )
        messages = [
            {"role": "system",
             "content": "You are a world-class resume writing expert. Your task is to refine resume content to be concise and perfectly tailored to a job description, while strictly respecting constraints on word count and layout to preserve the document's one-page format."},
            {"role": "user", "content": prompt}
        ]
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model="openai/gpt-4o-mini",
                    messages=messages,
                    temperature=0.6,
                    max_tokens=400,
                    top_p=1.0
                )
            )
            rewritten_text = response.choices[0].message.content.strip()
            # Clean up potential markdown code blocks from the response
            return re.sub(r'```.*?\n|```', '', rewritten_text, flags=re.DOTALL)

        except OpenAIError as e:
            logger.error(f"GitHub AI API error: {e}", exc_info=True)
            raise HTTPException(status_code=502, detail=f"An error occurred with the AI service: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in tailor_content: {e}", exc_info=True)
            return original_text  # Fallback to original text


def is_section_header(paragraph) -> bool:
    """Flexibly checks if a paragraph is a section header."""
    text = paragraph.text.strip()
    if not text or len(text) > 50:
        return False
    # Added "ABOUT" to the list of recognized headers
    header_pattern = re.compile(
        r'^(PROFESSIONAL EXPERIENCE|WORK HISTORY|EXPERIENCE|EDUCATION|SKILLS|OBJECTIVE|SUMMARY|ABOUT|TECHNICAL SKILLS|PROJECTS|ACHIEVEMENTS):?$',
        re.IGNORECASE)
    is_bold = any(run.bold for run in paragraph.runs)
    is_all_caps = text.isupper() and len(text.split()) < 5
    return bool(is_bold or is_all_caps or header_pattern.match(text))


def is_bullet_point(paragraph) -> bool:
    """Checks if a paragraph is a bullet point."""
    style_is_list = paragraph.style and paragraph.style.name.lower().startswith('list')
    text_is_bullet = paragraph.text.strip().startswith(("-", "•", "*"))
    return style_is_list or text_is_bullet


def replace_paragraph_text(paragraph, new_text):
    """
    Surgically replaces the text of a paragraph while preserving all formatting of its runs.
    """
    prefix = ""
    # Retain original bullet character if present
    if paragraph.text.strip().startswith('•'):
        prefix = '• '
    elif paragraph.text.strip().startswith('*'):
        prefix = '* '
    elif paragraph.text.strip().startswith('-'):
        prefix = '- '

    runs = paragraph.runs
    if not runs: return

    runs[0].text = prefix + new_text
    for i in range(1, len(runs)):
        runs[i].text = ''


async def process_docx_natively(docx_path: str, job_description: str) -> str:
    """
    Processes the DOCX using block-based logic for different section types.
    """
    doc = Document(docx_path)

    LOCKED_SECTIONS = ["EDUCATION", "ACHIEVEMENTS"]
    BLOCK_SECTIONS = ["OBJECTIVE", "SUMMARY", "ABOUT"]

    tasks = []
    task_metadata = []  # To map results back to the document structure

    # 1. Group document content into blocks by section header
    content_blocks = []
    current_block = None
    for para in doc.paragraphs:
        if is_section_header(para):
            if current_block:
                content_blocks.append(current_block)
            section_name = para.text.strip().upper().replace(':', '')
            current_block = {'section': section_name, 'paragraphs': []}
        elif current_block:
            current_block['paragraphs'].append(para)
    if current_block:  # Add the last block to the list
        content_blocks.append(current_block)

    # 2. Create AI tasks based on the content of each block
    for block in content_blocks:
        section = block['section']
        if any(locked in section for locked in LOCKED_SECTIONS):
            continue

        if any(block_sec in section for block_sec in BLOCK_SECTIONS):
            # Process Objective, Summary, etc. as a single block
            original_text = "\n".join([p.text for p in block['paragraphs']]).strip()
            if original_text:
                task = tailor_content(section, job_description, original_text)
                tasks.append(task)
                task_metadata.append({'type': 'block', 'block': block})
        else:
            # Process other sections (Experience, Projects) as individual bullet points
            for para in block['paragraphs']:
                text = para.text.strip()
                if text and is_bullet_point(para):
                    text_for_ai = re.sub(r'^[•*-]\s*', '', text)
                    task = tailor_content(section, job_description, text_for_ai)
                    tasks.append(task)
                    task_metadata.append({'type': 'bullet', 'para': para})

    if not tasks:
        logger.warning("No editable content was found. The document will not be changed.")
        return docx_path

    # 3. Execute all tasks concurrently
    logger.info(f"Found {len(tasks)} items to process. Sending to AI...")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("AI processing complete. Applying changes to the document...")

    # 4. Apply changes back to the document
    changes_applied = 0
    for i, res in enumerate(results):
        meta = task_metadata[i]
        if isinstance(res, Exception):
            logger.error(f"Task for section '{meta.get('block', {}).get('section', '')}' failed: {res}. Original kept.")
            continue

        if meta['type'] == 'block':
            # Replace a whole block of text (for Skills/Objective)
            original_text = "\n".join([p.text for p in meta['block']['paragraphs']]).strip()
            if res.strip().lower() != original_text.lower():
                logger.info(f"Applying BLOCK change for section '{meta['block']['section']}'")
                new_lines = res.strip().split('\n')
                block_paras = meta['block']['paragraphs']
                for j, para in enumerate(block_paras):
                    if j < len(new_lines):
                        replace_paragraph_text(para, new_lines[j])
                    else:  # Clear out extra paragraphs if the new text is shorter
                        para.clear()
                changes_applied += 1

        elif meta['type'] == 'bullet':
            # Replace a single bullet point
            para = meta['para']
            original_text = re.sub(r'^[•*-]\s*', '', para.text.strip())
            if res.strip().lower() != original_text.lower():
                logger.info(f"Applying change for bullet: {res[:50]}...")
                replace_paragraph_text(para, res)
                changes_applied += 1

    logger.info(f"Processing finished. Total changes applied: {changes_applied}/{len(tasks)}.")

    tailored_docx_path = docx_path.replace(".docx", "_tailored.docx")
    doc.save(tailored_docx_path)
    return tailored_docx_path


# --- API ENDPOINTS (No changes below this line) ---

@app.post("/tailor-resume")
async def tailor_resume_endpoint(job_description: str = Form(...), resume: UploadFile = File(...)):
    if not resume.filename.lower().endswith('.docx'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .docx file.")

    secure_suffix = str(uuid.uuid4())
    original_filename = os.path.splitext(resume.filename)[0].replace(" ", "_")
    docx_path = os.path.join(UPLOAD_DIR, f"{original_filename}_{secure_suffix}.docx")

    logger.info(f"Receiving DOCX file '{resume.filename}'. Saving as '{docx_path}'.")
    try:
        with open(docx_path, "wb") as f:
            shutil.copyfileobj(resume.file, f)

        tailored_docx_path = await process_docx_natively(docx_path, job_description)

        tailored_pdf_path = tailored_docx_path.replace(".docx", ".pdf")
        if tailored_docx_path != docx_path:
            logger.info(f"Converting {os.path.basename(tailored_docx_path)} to PDF...")
            convert(tailored_docx_path, tailored_pdf_path)
            logger.info("Conversion to PDF successful.")
        else:
            logger.info("No changes were made, converting original document to PDF.")
            convert(docx_path, tailored_pdf_path.replace("_tailored", ""))
            tailored_pdf_path = tailored_pdf_path.replace("_tailored", "")

        if os.path.exists(docx_path):
            os.remove(docx_path)

        return {
            "message": "Resume processing complete!",
            "pdf_download_url": f"/download/{os.path.basename(tailored_pdf_path)}",
            "docx_download_url": f"/download/{os.path.basename(tailored_docx_path)}",
        }
    except Exception as e:
        logger.error(f"An unexpected error occurred in tailor_resume_endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process the resume. Error: {e}")


@app.get("/download/{filename}")
async def download_file(filename: str, background_tasks: BackgroundTasks):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404,
                            detail=f"File not found: {filename}. It may have been deleted or never created.")

    background_tasks.add_task(os.remove, file_path)

    return FileResponse(path=file_path, filename=filename, media_type='application/octet-stream')


@app.get("/", summary="Health Check")
async def health_check():
    """Health check endpoint."""
    return {"status": "OK", "message": "Resume Tailoring API v7 (Block-Aware) is running."}

