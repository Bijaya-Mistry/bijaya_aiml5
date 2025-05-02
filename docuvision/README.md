# DocuVision 📄✨
A FastAPI-based document intelligence API that extracts embedded images and structured content from DOCX and PDF files, and uses GPT-4 to answer prompts with visual references.

---

## 🚀 Features

- 🔍 Extracts figures from DOCX and both scanned/text PDFs.
- 🧠 Classifies PDFs into text-based or image-based before processing.
- 🤖 Uses Azure OpenAI GPT-4o to answer prompts with attached images.
- 🖼️ Returns images as base64 so they can be rendered in web UIs.
- 🌐 API-first design with FastAPI.

---

## 🗂️ Project Structure

── app/
│ ├── main.py # FastAPI entry point
│ ├── processor.py # Core logic
│ ├── utils.py # Image extraction, text parsing
│ ├── schemas.py # Pydantic models
│ └── init.py
├── uploads/ # Uploaded files go here
├── .env # Your secrets (excluded from Git)
├── requirements.txt # All dependencies
├── README.md # You're reading it
└── .gitignore # Prevents accidental commits of sensitive or large files  
