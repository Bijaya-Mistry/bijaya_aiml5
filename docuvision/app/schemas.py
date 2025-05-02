from pydantic import BaseModel
from typing import List, Optional

class ImageData(BaseModel):
    image_path: str
    image_base64: str

class FileResponse(BaseModel):
    message: Optional[str] = None
    generated_response: Optional[str] = None
    images: Optional[List[ImageData]] = None
  
