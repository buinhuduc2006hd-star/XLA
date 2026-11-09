from pathlib import Path
from uuid import uuid4
import sys

sys.path.append(str(Path(__file__).resolve().parent))


from services.id_card_extractor import extract_id_card_fields

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="CCCD Reader API",
    description="Backend API for extracting information from Vietnamese CCCD images",
    version="0.1.0",
)

# Cấu hình CORS cho phép Frontend truy cập
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong thực tế sản xuất có thể đổi "*" thành domain FE của bạn
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


@app.get("/")
def root():
    return {
        "message": "CCCD Reader API is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok"
    }


@app.post("/api/v1/upload")
async def upload_image(file: UploadFile = File(...)):
    extension = Path(file.filename).suffix.lower() if file.filename else ""

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG and PNG images are allowed",
        )

    file_id = uuid4().hex
    filename = f"{file_id}{extension}"
    file_path = UPLOAD_DIR / filename

    content = await file.read()

    with open(file_path, "wb") as buffer:
        buffer.write(content)

    try:
        card_data = extract_id_card_fields(
            str(file_path)
        )
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )
    finally:
        # Tự động dọn dẹp file tạm sau khi OCR xử lý xong để tránh đầy bộ nhớ RAM/Disk
        file_path.unlink(missing_ok=True)

    return {
        "message": "CCCD extraction completed successfully",
        "filename": filename,
        "size": len(content),
        "data": card_data,
    }