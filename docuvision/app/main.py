from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from . import processor
from .schemas import FileResponse

import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/upload-file/", response_model=FileResponse)
async def upload_file(file: UploadFile = File(...), prompt: str = Form(None)):
    if not file.filename.lower().endswith((".docx", ".pdf")):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a DOCX or PDF file.")
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    return await processor.process_file(file_path, prompt)
  
