"""CCCD & GPLX Reader - Ứng dụng Desktop & Mobile viết bằng Flet (Giai đoạn 5).

Tích hợp End-to-End với FastAPI v2:
- Pipeline: YOLOv8n -> Document Rectification -> EasyOCR -> Field Validation.
- Hỗ trợ chế độ Fast Mode (ưu tiên tốc độ) và Accurate Mode (ưu tiên độ chính xác).
- Hỗ trợ tự động nhận diện loại giấy tờ (Auto) hoặc chỉ định CCCD / GPLX.
- Hiển thị ảnh xem trước, tiến trình không khóa giao diện, và cho phép thử lại.
- Các trường kết quả dạng TextField có thể chỉnh sửa trực tiếp, đánh dấu trường đã sửa.
- Hiển thị trạng thái validator và nguồn trích xuất dữ liệu, không sinh confidence giả.
- Xử lý mã lỗi chi tiết (400, 413, 422, 503, timeout, mất kết nối mạng).
- Tương thích ngược hoàn toàn với API v1 baseline.
- An toàn tuyệt đối trên Windows 11 (tự động vô hiệu hóa camera nếu không hỗ trợ).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import flet as ft
import flet_camera as fc
import flet_permission_handler as fph
import httpx

DEFAULT_API_URL: str = os.getenv("CCCD_API_URL", "http://127.0.0.1:8000/api/v2/upload")
DEFAULT_TIMEOUT: float = float(os.getenv("CCCD_API_TIMEOUT", "180.0"))
MAX_BYTES: int = 15 * 1024 * 1024  # 15 MB

ALLOWED_MIME: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

CARD_TITLES: dict[str, str] = {
    "cccd": "Căn cước công dân",
    "gplx": "Giấy phép lái xe",
}

FIELD_LABELS: dict[str, dict[str, str]] = {
    "cccd": {
        "id_number": "Số CCCD",
        "full_name": "Họ và tên",
        "date_of_birth": "Ngày sinh",
        "gender": "Giới tính",
        "nationality": "Quốc tịch",
        "place_of_origin": "Quê quán",
        "residence": "Nơi thường trú",
        "expiry_date": "Có giá trị đến",
    },
    "gplx": {
        "license_number": "Số GPLX",
        "full_name": "Họ và tên",
        "date_of_birth": "Ngày sinh",
        "nationality": "Quốc tịch",
        "address": "Nơi cư trú",
        "license_class": "Hạng",
        "expiry": "Có giá trị đến",
    },
}

SOURCE_LABELS: dict[str, str] = {
    "rectified": "Nắn phẳng",
    "pose_rectified": "Nắn phẳng (Pose)",
    "yolo_crop_baseline": "Crop YOLO",
    "failed": "Thất bại",
    "unextracted": "Chưa đọc được",
    "v1_baseline": "Fixed ROI (v1)",
}

ERROR_TRANSLATIONS: dict[str, str] = {
    "FILE_TOO_LARGE": "Dung lượng ảnh vượt quá giới hạn 15MB. Vui lòng nén hoặc chọn ảnh nhỏ hơn.",
    "IMAGE_TOO_LARGE": "Độ phân giải ảnh quá lớn (> 36 megapixels). Vui lòng chọn ảnh có kích thước phù hợp để tránh quá tải bộ nhớ.",
    "IMAGE_TOO_SMALL": "Kích thước ảnh quá nhỏ (tối thiểu 10x10 px). Vui lòng chọn ảnh rõ ràng hơn.",
    "INVALID_IMAGE": "File ảnh bị hỏng hoặc không đúng định dạng JPG, PNG, WEBP hợp lệ.",
    "UNSUPPORTED_MEDIA_TYPE": "Định dạng file không được hỗ trợ. Vui lòng tải ảnh JPG, PNG hoặc WEBP.",
    "NO_DOCUMENT_DETECTED": "Không phát hiện thấy giấy tờ (CCCD hoặc GPLX) trong ảnh. Vui lòng chụp rõ ràng, đủ sáng và lấy trọn vẹn thẻ.",
    "DOCUMENT_TYPE_MISMATCH": "Loại giấy tờ phát hiện không khớp với loại bạn đã chọn. Vui lòng chọn đúng loại hoặc để chế độ 'Tự động phát hiện'.",
    "MODEL_CONFIGURATION_ERROR": "Mô hình nhận diện trên máy chủ chưa được cấu hình hoặc thiếu trọng số huấn luyện. Vui lòng kiểm tra lại server.",
    "POSE_MODEL_NOT_FOUND": "Mô hình YOLOv8-Pose chưa có checkpoint trọng số (models/document_pose/best.pt). Vui lòng cung cấp checkpoint mô hình trên máy chủ.",
    "POSE_MODEL_INCOMPATIBLE": "Mô hình Pose trên server không tương thích hoặc thiếu cấu hình keypoint 4 góc.",
    "KEYPOINT_VALIDATION_FAILED": "Không thể xác định 4 góc vật lý an toàn của thẻ (thẻ bị che khuất, quá nghiêng hoặc góc bo bị đứt gãy). Vui lòng chụp lại rõ ràng hơn.",
    "EMPTY_FILE": "File ảnh rỗng (0 bytes). Vui lòng chọn file hợp lệ.",
    "READ_ERROR": "Lỗi đọc dữ liệu file tải lên từ thiết bị.",
    "INTERNAL_SERVER_ERROR": "Máy chủ gặp sự cố nội bộ trong quá trình xử lý ảnh.",
}


@dataclass
class State:
    image_bytes: bytes | None = None
    filename: str = ""
    # Giá trị OCR thô ban đầu nhận từ API
    raw_ocr_values: dict[str, str] = field(default_factory=dict)
    # Giá trị hiện tại (bao gồm các chỉnh sửa thủ công của người dùng)
    edited_values: dict[str, str] = field(default_factory=dict)
    # Cờ validation từ backend
    validation_status: dict[str, bool] = field(default_factory=dict)
    # Nguồn gốc trích xuất từng trường
    sources: dict[str, str] = field(default_factory=dict)
    # Metadata bổ sung từ v2
    doc_type: str = "cccd"
    detection_info: dict[str, Any] | None = None
    processing_info: dict[str, Any] | None = None
    timing_info: dict[str, Any] | None = None


def extract_error(response: httpx.Response) -> str:
    """Lấy thông điệp lỗi dễ hiểu cho người dùng từ response của API v1 hoặc v2."""
    try:
        body: object = response.json()
    except Exception:
        if response.status_code == 413:
            return "Dung lượng file tải lên quá lớn (HTTP 413). Vui lòng chọn ảnh dưới 15MB."
        if response.status_code == 503:
            return "Dịch vụ máy chủ tạm thời không khả dụng (HTTP 503). Vui lòng kiểm tra server."
        return f"Lỗi HTTP {response.status_code} từ máy chủ."

    if isinstance(body, dict):
        err_code = body.get("error_code")
        if err_code and err_code in ERROR_TRANSLATIONS:
            trans = ERROR_TRANSLATIONS[err_code]
            msg = body.get("message")
            if msg and msg != trans:
                return f"{trans}\n(Chi tiết: {msg})"
            return trans

        # Lỗi validation dạng danh sách của FastAPI
        detail: object = body.get("detail") or body.get("message")
        if isinstance(detail, str):
            return detail
        if isinstance(detail, list):
            parts: list[str] = []
            for item in detail:
                if isinstance(item, dict):
                    loc = ".".join(str(p) for p in item.get("loc", []))
                    parts.append(f"{loc}: {item.get('msg', '')}")
            if parts:
                return "\n".join(parts)

    return f"Lỗi {response.status_code} từ máy chủ."


def render_result_v2(
    payload: dict[str, Any],
    state: State,
    result_column: ft.Column,
    result_card: ft.Card,
    page: ft.Page | None = None,
    clipboard: ft.Clipboard | None = None,
) -> None:
    """Hiển thị kết quả API v2 hoặc adapter API v1 dưới dạng form có thể chỉnh sửa."""
    result_column.controls.clear()

    # 1. Phân loại response API v2 vs v1
    is_v2 = "fields" in payload
    if is_v2:
        doc_type = payload.get("document_type", "cccd")
        raw_fields = payload.get("fields", {})
        raw_ocr_dict = payload.get("raw_ocr", {})
        detection = payload.get("detection") or {}
        processing = payload.get("processing") or {}
        timing = payload.get("timing_ms") or {}
    else:
        # Adapter cho API v1 cũ
        data = payload.get("data", {})
        doc_type = data.get("card_type", "cccd")
        raw_fields = {}
        raw_ocr_dict = {}
        for k, val in data.items():
            if k == "card_type":
                continue
            raw_fields[k] = {
                "value": val,
                "raw_value": val,
                "valid": bool(val),
                "clean": True,
                "confidence": None,
                "source": "v1_baseline",
            }
        detection = {}
        processing = {}
        timing = {}

    state.doc_type = doc_type
    state.detection_info = detection
    state.processing_info = processing
    state.timing_info = timing
    state.raw_ocr_values.clear()
    state.edited_values.clear()
    state.validation_status.clear()
    state.sources.clear()

    # 2. Header thông tin nhận diện
    card_title = CARD_TITLES.get(doc_type, doc_type.upper())
    total_time_str = f"{timing.get('total_ms', 0) / 1000.0:.2f}s" if timing.get("total_ms") else ""
    yolo_conf = detection.get("confidence")
    conf_str = f" · YOLO: {yolo_conf * 100:.0f}%" if yolo_conf is not None else ""
    timing_str = f" · {total_time_str}" if total_time_str else ""

    title_row = ft.Row(
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        controls=[
            ft.Text(f"Kết quả nhận diện: {card_title}", size=16, weight=ft.FontWeight.BOLD),
            ft.Text(f"{conf_str}{timing_str}", size=11, color=ft.Colors.BLUE_GREY_500),
        ],
    )
    result_column.controls.append(title_row)

    # 3. Banner thông tin chi tiết (nếu là v2)
    if is_v2:
        rect_ok = processing.get("rectification_success", False)
        rect_str = "Nắn phối cảnh: Thành công" if rect_ok else "Nắn phối cảnh: Fallback YOLO"
        orient_deg = processing.get("orientation_degrees", 0)
        meta_chips = ft.Row(
            wrap=True,
            spacing=6,
            controls=[
                ft.Container(
                    content=ft.Text(rect_str, size=10, color=ft.Colors.BLUE_900),
                    bgcolor=ft.Colors.BLUE_50,
                    border_radius=ft.BorderRadius.all(4),
                    padding=ft.Padding.symmetric(horizontal=6, vertical=2),
                ),
                ft.Container(
                    content=ft.Text(f"Góc xoay: {orient_deg}°", size=10, color=ft.Colors.TEAL_900),
                    bgcolor=ft.Colors.TEAL_50,
                    border_radius=ft.BorderRadius.all(4),
                    padding=ft.Padding.symmetric(horizontal=6, vertical=2),
                ),
            ],
        )
        result_column.controls.append(meta_chips)

    result_column.controls.append(ft.Divider(height=1))

    # 4. Form các trường dữ liệu có thể chỉnh sửa
    labels_map = FIELD_LABELS.get(doc_type, {})

    for field_key, field_data in raw_fields.items():
        if isinstance(field_data, dict):
            val_init = str(field_data.get("value") or "")
            raw_val = str(field_data.get("raw_value") or raw_ocr_dict.get(field_key) or val_init)
            is_valid = bool(field_data.get("valid", False))
            is_clean = bool(field_data.get("clean", True))
            source = str(field_data.get("source") or "rectified")
        else:
            val_init = str(field_data or "")
            raw_val = str(raw_ocr_dict.get(field_key) or val_init)
            is_valid = bool(val_init)
            is_clean = True
            source = "rectified"

        # Phân biệt rõ dữ liệu OCR thô và giá trị hiển thị/chỉnh sửa
        state.raw_ocr_values[field_key] = raw_val
        state.edited_values[field_key] = val_init
        state.validation_status[field_key] = is_valid
        state.sources[field_key] = source

        field_label = labels_map.get(field_key, field_key)

        # Icon trạng thái validation
        if is_valid:
            valid_icon = ft.Icon(
                ft.Icons.CHECK_CIRCLE,
                color=ft.Colors.GREEN_600,
                size=18,
                tooltip="Định dạng hợp lệ theo chuẩn (đã làm sạch nhãn)",
            )
        else:
            valid_icon = ft.Icon(
                ft.Icons.WARNING_AMBER_ROUNDED,
                color=ft.Colors.AMBER_700,
                size=18,
                tooltip="Chưa khớp định dạng chuẩn hoặc còn dính nhãn — Vui lòng kiểm tra lại",
            )

        # Badge nguồn trích xuất
        source_txt = SOURCE_LABELS.get(source, source)
        src_badge = ft.Container(
            content=ft.Text(source_txt, size=10, color=ft.Colors.BLUE_GREY_700),
            bgcolor=ft.Colors.BLUE_GREY_50,
            border=ft.Border.all(1, ft.Colors.BLUE_GREY_200),
            border_radius=ft.BorderRadius.all(4),
            padding=ft.Padding.symmetric(horizontal=6, vertical=2),
        )

        # Badge thông báo "Đã chỉnh sửa"
        edit_badge = ft.Container(
            content=ft.Text("Đã sửa", size=10, weight=ft.FontWeight.BOLD, color=ft.Colors.AMBER_900),
            bgcolor=ft.Colors.AMBER_100,
            border_radius=ft.BorderRadius.all(4),
            padding=ft.Padding.symmetric(horizontal=5, vertical=2),
            visible=False,
        )

        # Input TextField cho phép sửa
        def make_change_handler(fk: str, badge: ft.Container, init_v: str):
            def handler(e: ft.ControlEvent):
                new_val = e.control.value or ""
                state.edited_values[fk] = new_val
                badge.visible = (new_val != init_v)
                if page:
                    page.update()
            return handler

        txt_input = ft.TextField(
            value=val_init,
            dense=True,
            text_size=13,
            expand=True,
            hint_text="Chưa nhận diện được",
            on_change=make_change_handler(field_key, edit_badge, val_init),
        )

        field_control = ft.Column(
            spacing=3,
            controls=[
                ft.Row(
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    controls=[
                        ft.Text(field_label, size=12, weight=ft.FontWeight.W_500, color=ft.Colors.BLUE_GREY_700),
                        ft.Row([edit_badge, src_badge, valid_icon], spacing=4),
                    ],
                ),
                txt_input,
            ],
        )
        result_column.controls.append(field_control)

    # 5. Thanh nút chức năng: Sao chép JSON / Xuất thông tin
    async def on_copy_json(_: ft.ControlEvent):
        export_payload = {
            "document_type": state.doc_type,
            "fields": state.edited_values,
            "raw_ocr": state.raw_ocr_values,
            "validation": state.validation_status,
        }
        json_str = json.dumps(export_payload, ensure_ascii=False, indent=2)
        if clipboard:
            await clipboard.set(json_str)
        if page:
            snack = ft.SnackBar(
                content=ft.Text("Đã sao chép toàn bộ thông tin JSON vào Clipboard!"),
                open=True,
            )
            page.overlay.append(snack)
            page.update()

    copy_btn = ft.Button(
        "Sao chép JSON",
        icon=ft.Icons.COPY_ALL,
        on_click=on_copy_json,
    )

    result_column.controls.append(ft.Divider(height=10))
    result_column.controls.append(
        ft.Row([copy_btn], alignment=ft.MainAxisAlignment.END)
    )

    result_card.visible = True
    if page:
        page.update()


async def main(page: ft.Page) -> None:
    page.title = "CCCD & GPLX Reader"
    page.padding = ft.Padding.all(16)
    page.scroll = ft.ScrollMode.AUTO
    page.theme_mode = ft.ThemeMode.LIGHT
    page.horizontal_alignment = ft.CrossAxisAlignment.STRETCH

    state = State()

    # Khởi tạo các services
    file_picker = ft.FilePicker()
    page.services.append(file_picker)

    clipboard = ft.Clipboard()
    page.services.append(clipboard)

    is_mobile: bool = page.platform in (ft.PagePlatform.ANDROID, ft.PagePlatform.IOS)
    can_use_camera: bool = page.web or is_mobile

    # Xin quyền camera lúc chạy (bắt buộc trên Android/iOS)
    permission_handler: fph.PermissionHandler | None = (
        fph.PermissionHandler() if is_mobile else None
    )
    if permission_handler is not None:
        page.services.append(permission_handler)

    # ---------- Controls cấu hình ----------
    api_url = ft.TextField(
        label="API Endpoint URL",
        value=DEFAULT_API_URL,
        text_size=12,
        dense=True,
        helper="Hỗ trợ cả /api/v2/upload (mới) và /api/v1/upload (cũ).",
    )

    mode_dropdown = ft.Dropdown(
        label="Chế độ nhận diện (Mode)",
        value="fast",
        options=[
            ft.dropdown.Option("fast", "Nhanh (Fast Mode) — Khuyên dùng"),
            ft.dropdown.Option("accurate", "Chính xác cao (Accurate Mode) — Bounded Retry"),
        ],
        dense=True,
        text_size=12,
        expand=True,
    )

    doc_type_dropdown = ft.Dropdown(
        label="Loại giấy tờ (Document Type)",
        value="auto",
        options=[
            ft.dropdown.Option("auto", "Tự động phát hiện (Auto Detect)"),
            ft.dropdown.Option("cccd", "Căn cước công dân (CCCD)"),
            ft.dropdown.Option("gplx", "Giấy phép lái xe (GPLX)"),
        ],
        dense=True,
        text_size=12,
        expand=True,
    )

    options_row = ft.Row([mode_dropdown, doc_type_dropdown], spacing=10)

    # Ảnh 1x1 trong suốt làm placeholder
    placeholder = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06"
        b"\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe"
        b"\x02\xfe\xa7\x9a\xa0\xa0\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    preview_image = ft.Image(src=placeholder, height=280, fit=ft.BoxFit.CONTAIN)
    file_info = ft.Text(size=12, color=ft.Colors.BLUE_GREY_500)
    preview_box = ft.Column(
        visible=False,
        controls=[preview_image, file_info],
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
    )

    error_text = ft.Text(color=ft.Colors.RED_700, visible=False, selectable=True)
    status_label = ft.Text(size=12, italic=True, color=ft.Colors.BLUE_GREY_600, visible=False)
    progress = ft.ProgressRing(width=22, height=22, visible=False)
    reset_btn = ft.OutlinedButton("Xoá ảnh", icon=ft.Icons.DELETE_OUTLINE, visible=False)
    retry_btn = ft.Button("Thử lại", icon=ft.Icons.REFRESH, visible=False)

    result_column = ft.Column(spacing=10)
    result_card = ft.Card(visible=False, content=ft.Container(padding=ft.Padding.all(16), content=result_column))

    # Control Camera an toàn
    camera: fc.Camera | None = (
        fc.Camera(expand=True, preview_enabled=True) if can_use_camera else None
    )
    take_btn = ft.Button("Chụp", icon=ft.Icons.PHOTO_CAMERA)
    cancel_cam_btn = ft.OutlinedButton("Huỷ")
    camera_panel = ft.Column(
        visible=False,
        controls=(
            [
                ft.Container(
                    height=420,
                    bgcolor=ft.Colors.BLACK,
                    border_radius=ft.BorderRadius.all(8),
                    content=camera,
                ),
                ft.Row([take_btn, cancel_cam_btn], alignment=ft.MainAxisAlignment.CENTER),
            ]
            if camera is not None
            else []
        ),
    )

    pick_btn = ft.Button("Chọn ảnh", icon=ft.Icons.FOLDER_OPEN)
    camera_btn = ft.Button("Chụp ảnh", icon=ft.Icons.PHOTO_CAMERA, visible=can_use_camera)

    # ---------- Helpers ----------
    def set_error(message: str | None) -> None:
        error_text.value = message or ""
        error_text.visible = bool(message)
        retry_btn.visible = bool(message and state.image_bytes is not None)

    def set_image(data: bytes, name: str) -> bool:
        """Lưu ảnh vào state và hiện preview. Trả về False nếu ảnh không hợp lệ."""
        set_error(None)
        result_card.visible = False
        retry_btn.visible = False

        if Path(name).suffix.lower() not in ALLOWED_MIME:
            set_error("Chỉ chấp nhận ảnh JPG, JPEG, PNG hoặc WEBP.")
            return False
        if not data:
            set_error("File ảnh rỗng.")
            return False
        if len(data) > MAX_BYTES:
            set_error(f"Ảnh quá lớn (tối đa {MAX_BYTES // (1024 * 1024)}MB).")
            return False

        state.image_bytes = data
        state.filename = name
        preview_image.src = data
        file_info.value = f"{name} · {len(data) / 1024:.0f} KB"
        preview_box.visible = True
        reset_btn.visible = True
        return True

    # ---------- Handlers ----------
    async def on_pick(_: ft.ControlEvent) -> None:
        files = await file_picker.pick_files(
            file_type=ft.FilePickerFileType.IMAGE,
            with_data=True,
        )
        if not files:
            return
        picked = files[0]
        data: bytes | None = picked.bytes
        if data is None and picked.path:
            data = Path(picked.path).read_bytes()
        if data is None:
            set_error("Không đọc được file đã chọn.")
            page.update()
        elif set_image(data, picked.name):
            await process_image()  # Tự động nhận diện ngay sau khi chọn ảnh
        else:
            page.update()

    async def open_camera(_: ft.ControlEvent) -> None:
        if camera is None:
            return
        if permission_handler is not None:
            status = await permission_handler.request(fph.Permission.CAMERA)
            if status != fph.PermissionStatus.GRANTED:
                set_error("Cần cấp quyền Camera cho app trong Cài đặt để chụp ảnh.")
                page.update()
                return
        set_error(None)
        main_panel.visible = False
        camera_panel.visible = True
        page.update()
        try:
            cameras = await camera.get_available_cameras()
            if not cameras:
                raise RuntimeError("Không tìm thấy camera.")
            chosen = next(
                (c for c in cameras if c.lens_direction == fc.CameraLensDirection.BACK),
                cameras[0],
            )
            await camera.initialize(
                description=chosen,
                resolution_preset=fc.ResolutionPreset.HIGH,
                enable_audio=False,
                image_format_group=fc.ImageFormatGroup.JPEG,
            )
        except Exception as ex:
            close_camera()
            set_error(f"Không mở được camera: {ex}")
            page.update()

    def close_camera(_: object = None) -> None:
        camera_panel.visible = False
        main_panel.visible = True
        page.update()

    async def take_photo(_: ft.ControlEvent) -> None:
        if camera is None:
            return
        try:
            data = await camera.take_picture()
        except Exception as ex:
            close_camera()
            set_error(f"Không chụp được ảnh: {ex}")
            page.update()
            return
        close_camera()
        if set_image(data, "capture.jpg"):
            await process_image()
        else:
            page.update()

    def on_reset(_: ft.ControlEvent) -> None:
        state.image_bytes = None
        state.filename = ""
        preview_box.visible = False
        reset_btn.visible = False
        retry_btn.visible = False
        result_card.visible = False
        set_error(None)
        page.update()

    async def process_image(_: ft.ControlEvent | None = None) -> None:
        if state.image_bytes is None:
            return

        set_error(None)
        result_card.visible = False
        retry_btn.visible = False
        progress.visible = True
        status_label.value = "Đang nhận diện qua YOLOv8 & EasyOCR... Vui lòng đợi trong giây lát."
        status_label.visible = True

        # Ngăn người dùng gửi nhiều request cùng lúc (Concurrency Safety)
        pick_btn.disabled = True
        camera_btn.disabled = True
        reset_btn.disabled = True
        page.update()

        target_url = (api_url.value or "").strip()
        mime = ALLOWED_MIME.get(Path(state.filename).suffix.lower(), "image/jpeg")

        is_v2_endpoint = "/api/v2" in target_url
        form_data: dict[str, str] = {}
        if is_v2_endpoint:
            form_data["mode"] = mode_dropdown.value or "fast"
            form_data["document_type"] = doc_type_dropdown.value or "auto"

        try:
            # Sử dụng timeout 180s cho các pipeline AI OCR phức tạp
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                response = await client.post(
                    target_url,
                    files={"file": (state.filename, state.image_bytes, mime)},
                    data=form_data if is_v2_endpoint else None,
                )

            if response.status_code != 200:
                set_error(extract_error(response))
            else:
                payload = response.json()
                render_result_v2(payload, state, result_column, result_card, page, clipboard)

        except httpx.TimeoutException:
            set_error(
                f"Yêu cầu xử lý vượt quá thời gian cho phép ({DEFAULT_TIMEOUT:.0f}s). "
                "Máy chủ có thể đang tải nặng hoặc CPU đang bận. Bạn có thể nhấn 'Thử lại' hoặc chọn Fast Mode."
            )
        except (httpx.ConnectError, httpx.NetworkError):
            set_error(
                f"Không thể kết nối đến máy chủ tại: {target_url}\n"
                "Vui lòng đảm bảo backend FastAPI đang chạy và địa chỉ URL là chính xác."
            )
        except Exception as ex:
            set_error(f"Lỗi không xác định khi gửi yêu cầu: {ex}")
        finally:
            progress.visible = False
            status_label.visible = False
            pick_btn.disabled = False
            camera_btn.disabled = False
            reset_btn.disabled = False
            page.update()

    pick_btn.on_click = on_pick
    camera_btn.on_click = open_camera
    take_btn.on_click = take_photo
    cancel_cam_btn.on_click = close_camera
    reset_btn.on_click = on_reset
    retry_btn.on_click = process_image

    # ---------- Layout tổng thể ----------
    main_panel = ft.Column(
        spacing=12,
        controls=[
            ft.Text("Đọc thông tin giấy tờ (CCCD / GPLX)", size=22, weight=ft.FontWeight.BOLD),
            ft.Text(
                "Chọn hoặc chụp ảnh CCCD/GPLX, hệ thống sẽ tự động phát hiện, nắn phối cảnh và trích xuất thông tin chuẩn xác.",
                size=13,
                color=ft.Colors.BLUE_GREY_600,
            ),
            api_url,
            options_row,
            ft.Row([pick_btn, camera_btn], wrap=True),
            preview_box,
            ft.Row([reset_btn, retry_btn, progress, status_label], wrap=True, alignment=ft.MainAxisAlignment.START),
            error_text,
            result_card,
        ],
    )

    page.add(ft.SafeArea(content=ft.Column([main_panel, camera_panel])))
    page.update()


if __name__ == "__main__":
    ft.run(main)
