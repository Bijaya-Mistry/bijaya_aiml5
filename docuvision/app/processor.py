import os
from .utils import (
    extract_images_from_docx,
    classify_pdf,
    extract_images_from_pdf,
    extract_images_from_scan_pdf,
    extract_relevant_sections,
    extract_file_paths,
    encode_image_to_base64,
    get_openai_client
)
from .schemas import FileResponse
from fastapi import HTTPException

async def process_file(file_path: str, prompt: str = None) -> FileResponse:
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".docx":
        images_dict, full_text = extract_images_from_docx(file_path)
    elif ext == ".pdf":
        classification = classify_pdf(file_path)
        if classification == "Text-based PDF":
            images_dict, full_text = extract_images_from_pdf(file_path)
        else:
            images_dict, full_text = extract_images_from_scan_pdf(file_path)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format.")

    if not prompt:
        return FileResponse(message="Prompt not provided.")

    relevant_text = extract_relevant_sections(full_text, prompt)
    if not relevant_text:
        return FileResponse(message="No relevant content found based on the prompt.")

    image_paths = extract_file_paths(relevant_text)
    image_data_list = []

    for path in image_paths:
        base64_img = encode_image_to_base64(path)
        if base64_img:
            image_data_list.append({"image_path": path, "image_base64": f"data:image/png;base64,{base64_img}"})
            relevant_text = relevant_text.replace(f'Fig_path= "{path}"', "[IMAGE ATTACHED]")

    try:
        messages = [
            {"role": "system", "content": "You are a knowledge search bot. Generate an appropriate response to the user's question."},
            {"role": "user", "content": f"Here are the RETRIEVED CHUNKS: {relevant_text}. Here is the QUESTION: {prompt}."}
        ]

        for img in image_data_list:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "[Attached Image]"},
                    {"type": "image_url", "image_url": {"url": img["image_base64"]}}
                ]
            })

        client = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=500
        )
        return FileResponse(generated_response=response.choices[0].message.content, images=image_data_list)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {str(e)}")
