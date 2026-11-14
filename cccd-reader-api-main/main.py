from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
from pathlib import Path
from uuid import uuid4
import sys

# Đảm bảo đường dẫn project nằm trong sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import io
import cv2
import numpy as np
from PIL import Image

# Giới hạn OpenCV dùng 1 luồng để tránh cạnh tranh tài nguyên CPU với PyTorch / EasyOCR
cv2.setNumThreads(1)
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from schemas import DocumentV2Response, ErrorResponse
from services.document_orientation import get_shared_easyocr_reader
from services.id_card_extractor import extract_id_card_fields
from services.pipeline_v2 import (
    DocumentTypeMismatchError,
    InvalidImageError,
    ModelConfigurationError,
    NoDocumentDetectedError,
    PipelineV2Error,
    get_document_detector,
    process_document_v2,
)

logger = logging.getLogger(__name__)

# Giới hạn kích thước file và ảnh
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB
MAX_IMAGE_PIXELS = 36_000_000           # 36 Megapixels (vd: 6000x6000) chống Decompression Bomb
MIN_IMAGE_DIMENSION = 10                # 10x10 px tối thiểu

ALLOWED_EXTENSIONS_V1 = {".jpg", ".jpeg", ".png"}
ALLOWED_EXTENSIONS_V2 = {".jpg", ".jpeg", ".png", ".webp"}

# Semaphore kiểm soát xử lý suy luận đồng thời trên CPU (tránh cạnh tranh CPU/RAM).
# Kiến trúc Concurrency Audit:
# 1. Semaphore(1) tuần tự hóa các tác vụ suy luận nặng trên CPU (YOLO + EasyOCR), ngăn ngừa tràn RAM trên 8GB RAM.
# 2. Xử lý qua asyncio.to_thread() giải phóng Event Loop, đảm bảo /health và /docs luôn phản hồi tức thì (< 5ms).
# 3. Sử dụng 'async with' đảm bảo semaphore luôn được giải phóng an toàn ngay cả khi OCR phát sinh exception.
# 4. Khi client ngắt kết nối/timeout, coroutine bị cancel thì async with context manager vẫn tự động giải phóng semaphore.
# 5. Lưu ý Production: Semaphore(1) nằm trong RAM của 1 process Python; nếu chạy Uvicorn multi-worker (--workers > 1)
#    sẽ phá vỡ cơ chế này và gây quá tải CPU/RAM. Do đó Uvicorn BẮT BUỘC chạy với --workers 1 trên môi trường CPU 8GB RAM.
_INFERENCE_SEMAPHORE = asyncio.Semaphore(1)


def _detect_and_validate_magic_bytes(content: bytes) -> str | None:
    """Xác thực định dạng ảnh qua magic bytes nhị phân thực tế (không tin cậy mime/extension client gửi)."""
    if len(content) < 12:
        return None
    if content.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if content.startswith(b"RIFF") and content[8:12] == b"WEBP":
        return "webp"
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý vòng đời ứng dụng: nạp trước mô hình một lần khi khởi động."""
    import os
    if not os.environ.get("DISABLE_MODEL_PRELOAD"):
        logger.info("CCCD Reader API đang khởi động. Khởi tạo tài nguyên mô hình...")
        try:
            # Pre-load YOLOv8n detector
            get_document_detector()
        except Exception as e:
            logger.warning(f"Chưa thể nạp trước YOLO detector khi khởi động: {e}")

        try:
            # Pre-load EasyOCR reader
            get_shared_easyocr_reader()
        except Exception as e:
            logger.warning(f"Chưa thể nạp trước EasyOCR reader khi khởi động: {e}")

    yield
    logger.info("CCCD Reader API đang tắt. Giải phóng tài nguyên...")


app = FastAPI(
    title="CCCD & GPLX Reader API",
    description="Backend API trích xuất thông tin Căn cước công dân và Giấy phép lái xe Việt Nam",
    version="0.2.0",
    lifespan=lifespan,
)

# Cấu hình CORS cho phép Frontend truy cập
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def root():
    return {
        "message": "CCCD & GPLX Reader API is running",
        "version": "0.2.0",
        "docs_url": "/docs",
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "version": "0.2.0",
    }


# ===========================================================================
# ENDPOINT V1 (GIỮ NGUYÊN KHẢ NĂNG HOẠT ĐỘNG BAN ĐẦU - GIAI ĐOẠN 1)
# ===========================================================================

@app.post("/api/v1/upload")
async def upload_image(file: UploadFile = File(...)):
    """Endpoint V1: Fixed ROI + EasyOCR (Baseline ban đầu)."""
    extension = Path(file.filename).suffix.lower() if file.filename else ""

    if extension not in ALLOWED_EXTENSIONS_V1:
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
        card_data = extract_id_card_fields(str(file_path))
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


# ===========================================================================
# ENDPOINT V2 (GIAI ĐOẠN 4: YOLOV8 + RECTIFIER + ORIENTATION + ADAPTIVE OCR)
# ===========================================================================

@app.post(
    "/api/v2/upload",
    response_model=DocumentV2Response,
    responses={
        200: {"model": DocumentV2Response, "description": "Trích xuất thông tin thành công"},
        400: {"model": ErrorResponse, "description": "Ảnh lỗi, rỗng hoặc tham số không hợp lệ"},
        413: {"model": ErrorResponse, "description": "Dung lượng file vượt quá giới hạn 15MB"},
        422: {"model": ErrorResponse, "description": "Không tìm thấy giấy tờ hoặc loại giấy tờ không khớp"},
        500: {"model": ErrorResponse, "description": "Lỗi nội bộ server"},
    },
)
async def upload_image_v2(
    file: UploadFile = File(..., description="File ảnh CCCD hoặc GPLX (JPG, PNG, WEBP)"),
    mode: str = Form("fast", description="Chế độ: 'fast' (nhanh, 1 pass) hoặc 'accurate' (tối ưu độ chính xác và fallback)"),
    document_type: str = Form("auto", description="Loại giấy tờ: 'auto' (tự phát hiện), 'cccd', hoặc 'gplx'"),
):
    """Endpoint V2: Pipeline tự động End-to-End từ phát hiện, nắn phối cảnh đến trích xuất OCR."""
    # 1. Kiểm tra phần mở rộng file
    filename = file.filename or ""
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS_V2:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                error_code="UNSUPPORTED_MEDIA_TYPE",
                message=f"Only JPG, JPEG, PNG, and WEBP image formats are supported. Got '{extension}'",
            ).model_dump(),
        )

    # 2. Đọc nội dung file theo chunk để tránh nạp toàn bộ file quá giới hạn vào RAM
    chunks: list[bytes] = []
    total_bytes = 0
    chunk_size = 64 * 1024  # 64 KB
    is_too_large = False

    try:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > MAX_FILE_SIZE_BYTES:
                is_too_large = True
                del chunks
                break
            chunks.append(chunk)
    except Exception as e:
        logger.error(f"Lỗi khi đọc file upload: {type(e).__name__}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                error_code="READ_ERROR",
                message="Failed to read uploaded file",
            ).model_dump(),
        )

    # 3. Kiểm tra dung lượng
    if is_too_large:
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content=ErrorResponse(
                error_code="FILE_TOO_LARGE",
                message=f"File size exceeds maximum allowed limit of {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB",
                details={"file_size_bytes": total_bytes, "max_bytes": MAX_FILE_SIZE_BYTES},
            ).model_dump(),
        )

    if total_bytes == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                error_code="EMPTY_FILE",
                message="Uploaded file is empty (0 bytes)",
            ).model_dump(),
        )

    content = b"".join(chunks)

    # 4. Kiểm tra magic bytes thực tế để xác thực định dạng ảnh (không tin cậy extension/MIME)
    detected_format = _detect_and_validate_magic_bytes(content)
    if detected_format is None:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                error_code="INVALID_IMAGE",
                message="File content does not match any allowed image format (JPEG, PNG, WEBP) magic bytes",
            ).model_dump(),
        )

    # 5. Kiểm tra metadata kích thước ảnh bằng Pillow trước khi giải mã mảng BGR (Chống Decompression Bomb)
    try:
        with Image.open(io.BytesIO(content)) as pil_img:
            w, h = pil_img.size
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                error_code="INVALID_IMAGE",
                message="Failed to parse image header. File may be corrupted or not a valid image format",
            ).model_dump(),
        )

    if h < MIN_IMAGE_DIMENSION or w < MIN_IMAGE_DIMENSION:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                error_code="IMAGE_TOO_SMALL",
                message=f"Image dimensions ({w}x{h}) are too small. Minimum required: {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION}",
            ).model_dump(),
        )

    total_pixels = h * w
    if total_pixels > MAX_IMAGE_PIXELS:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                error_code="IMAGE_TOO_LARGE",
                message=f"Image resolution ({w}x{h} = {total_pixels} pixels) exceeds safety limit of {MAX_IMAGE_PIXELS} pixels",
            ).model_dump(),
        )

    # 6. Giải mã ảnh trực tiếp trong RAM (chỉ thực hiện sau khi đã kiểm tra an toàn metadata)
    np_buf = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)

    if image is None:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                error_code="INVALID_IMAGE",
                message="Failed to decode image into pixel matrix",
            ).model_dump(),
        )

    # 6. Kiểm tra tham số mode và document_type
    mode_clean = str(mode).lower().strip()
    if mode_clean not in ("fast", "accurate"):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                error_code="INVALID_PARAMETER",
                message=f"Invalid mode: '{mode}'. Allowed values are 'fast' or 'accurate'",
            ).model_dump(),
        )

    doc_type_clean = str(document_type).lower().strip()
    if doc_type_clean not in ("auto", "cccd", "gplx"):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                error_code="INVALID_PARAMETER",
                message=f"Invalid document_type: '{document_type}'. Allowed values are 'auto', 'cccd', or 'gplx'",
            ).model_dump(),
        )

    # 7. Xử lý qua Pipeline V2 với Semaphore concurrency control và non-blocking thread
    try:
        async with _INFERENCE_SEMAPHORE:
            response_data = await asyncio.to_thread(
                process_document_v2,
                image=image,
                mode=mode_clean,
                document_type=doc_type_clean,
            )
        return response_data

    except NoDocumentDetectedError as e:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                error_code=e.error_code,
                message=e.message,
                details=e.details,
            ).model_dump(),
        )

    except DocumentTypeMismatchError as e:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                error_code=e.error_code,
                message=e.message,
                details=e.details,
            ).model_dump(),
        )

    except (InvalidImageError, PipelineV2Error) as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                error_code=e.error_code,
                message=e.message,
                details=e.details,
            ).model_dump(),
        )

    except ModelConfigurationError as e:
        logger.error(f"Lỗi cấu hình mô hình YOLO: {type(e).__name__}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(
                error_code="MODEL_CONFIGURATION_ERROR",
                message="YOLO document detection model is not properly configured or missing",
                details={"error": e.message if hasattr(e, "message") else str(e)},
            ).model_dump(),
        )

    except Exception as e:
        # Không rò rỉ stack trace hay thông tin nhạy cảm ra ngoài client
        logger.error(f"Lỗi không xác định trong quá trình xử lý Pipeline V2: {type(e).__name__}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error_code="INTERNAL_SERVER_ERROR",
                message="An unexpected internal error occurred during document processing",
            ).model_dump(),
        )
