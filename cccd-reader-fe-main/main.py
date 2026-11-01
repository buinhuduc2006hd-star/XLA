"""CCCD Reader - app đa nền tảng (Windows/Linux/macOS/Android/iOS) viết bằng Flet.

Chạy thử trên máy tính:   flet run main.py
Build APK:                flet build apk
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import flet as ft
import flet_camera as fc
import flet_permission_handler as fph
import httpx

DEFAULT_API_URL: str = os.getenv("CCCD_API_URL", "http://127.0.0.1:8000/api/v1/upload")
REQUEST_TIMEOUT: float = 60.0
MAX_BYTES: int = 10 * 1024 * 1024

ALLOWED_MIME: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}

CARD_TITLES: dict[str, str] = {
    "cccd": "Căn cước công dân",
    "gplx": "Giấy phép lái xe",
}

FIELD_LABELS: dict[str, dict[str, str]] = {
    "cccd": {
        "full_name": "Họ và tên",
        "date_of_birth": "Ngày sinh",
        "gender": "Giới tính",
        "nationality": "Quốc tịch",
        "place_of_origin": "Quê quán",
        "residence": "Nơi thường trú",
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
@dataclass
class State:
    image_bytes: bytes | None = None
    filename: str = ""


def extract_error(response: httpx.Response) -> str:
    """Lấy thông điệp lỗi dễ đọc từ response của API (kể cả 422 của FastAPI)."""
    try:
        body: object = response.json()
    except ValueError:
        return f"Lỗi {response.status_code} từ server."

    detail: object = body.get("detail") if isinstance(body, dict) else None
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
    return f"Lỗi {response.status_code} từ server."


async def main(page: ft.Page) -> None:
    page.title = "CCCD Reader"
    page.padding = 16
    page.scroll = ft.ScrollMode.AUTO
    page.theme_mode = ft.ThemeMode.LIGHT
    page.horizontal_alignment = ft.CrossAxisAlignment.STRETCH

    state = State()

    file_picker = ft.FilePicker()
    page.services.append(file_picker)

    is_mobile: bool = page.platform in (ft.PagePlatform.ANDROID, ft.PagePlatform.IOS)
    can_use_camera: bool = page.web or is_mobile

    # Xin quyền camera lúc chạy (bắt buộc trên Android/iOS trước khi mở camera)
    permission_handler: fph.PermissionHandler | None = (
        fph.PermissionHandler() if is_mobile else None
    )
    if permission_handler is not None:
        page.services.append(permission_handler)

    # ---------- Controls ----------
    api_url = ft.TextField(
        label="API URL",
        value=DEFAULT_API_URL,
        text_size=12,
        dense=True,
        helper="Trên điện thoại hãy dùng địa chỉ máy chủ thật (không phải 127.0.0.1).",
    )

    # Ảnh 1x1 trong suốt làm placeholder (Image yêu cầu có src khi khởi tạo)
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
    progress = ft.ProgressRing(width=20, height=20, visible=False)
    reset_btn = ft.OutlinedButton("Xoá ảnh", icon=ft.Icons.DELETE_OUTLINE, visible=False)

    result_column = ft.Column(spacing=6)
    result_card = ft.Card(visible=False, content=ft.Container(padding=16, content=result_column))

    # Control Camera chỉ hỗ trợ Android/iOS/Web: trên desktop không được tạo ra
    # (kể cả khi đang ẩn, Flet vẫn ném FletUnsupportedPlatformException).
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
                    border_radius=8,
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

    def set_image(data: bytes, name: str) -> bool:
        """Lưu ảnh vào state và hiện preview. Trả về False nếu ảnh không hợp lệ."""
        set_error(None)
        result_card.visible = False

        if Path(name).suffix.lower() not in ALLOWED_MIME:
            set_error("Chỉ chấp nhận ảnh JPG, JPEG hoặc PNG.")
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

    def render_result(data: dict[str, object]) -> None:
        result_column.controls = [ft.Text("Kết quả", size=16, weight=ft.FontWeight.BOLD)]
        for key, value in data.items():
            result_column.controls.append(
                ft.Row(
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    controls=[
                        ft.Text(FIELD_LABELS.get(key, key), color=ft.Colors.BLUE_GREY_500),
                        ft.Text(
                            str(value) if value else "Không đọc được",
                            weight=ft.FontWeight.W_600 if value else None,
                            italic=not value,
                            selectable=True,
                        ),
                    ],
                )
            )
        result_card.visible = True

    # ---------- Handlers ----------
    async def on_pick(_: ft.Event[ft.Button]) -> None:
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
            await process_image()  # tự động nhận diện ngay sau khi chọn ảnh
        else:
            page.update()

    async def open_camera(_: ft.Event[ft.Button]) -> None:
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
        except Exception as ex:  # plugin camera ném lỗi riêng theo từng nền tảng
            close_camera()
            set_error(f"Không mở được camera: {ex}")
            page.update()

    def close_camera(_: object = None) -> None:
        camera_panel.visible = False
        main_panel.visible = True
        page.update()

    async def take_photo(_: ft.Event[ft.Button]) -> None:
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

    def on_reset(_: ft.Event[ft.OutlinedButton]) -> None:
        state.image_bytes = None
        state.filename = ""
        preview_box.visible = False
        reset_btn.visible = False
        result_card.visible = False
        set_error(None)
        page.update()

    async def process_image() -> None:
        if state.image_bytes is None:
            return

        set_error(None)
        result_card.visible = False
        progress.visible = True
        pick_btn.disabled = True
        page.update()

        mime = ALLOWED_MIME[Path(state.filename).suffix.lower()]
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                response = await client.post(
                    (api_url.value or "").strip(),
                    files={"file": (state.filename, state.image_bytes, mime)},
                )
            if response.status_code != 200:
                set_error(extract_error(response))
            else:
                payload: object = response.json()
                data = payload.get("data") if isinstance(payload, dict) else None
                render_result(data if isinstance(data, dict) else {})
        except httpx.HTTPError:
            set_error("Không kết nối được tới API. Kiểm tra lại API URL và mạng.")
        finally:
            progress.visible = False
            pick_btn.disabled = False
            page.update()

    pick_btn.on_click = on_pick
    camera_btn.on_click = open_camera
    take_btn.on_click = take_photo
    cancel_cam_btn.on_click = close_camera
    reset_btn.on_click = on_reset

    # ---------- Layout ----------
    main_panel = ft.Column(
        spacing=12,
        controls=[
            ft.Text("Đọc thông tin giấy tờ (CCCD / GPLX)", size=22, weight=ft.FontWeight.BOLD),
            ft.Text(
                "Chọn hoặc chụp ảnh giấy tờ, app sẽ tự nhận diện và trích xuất thông tin.",
                size=13,
                color=ft.Colors.BLUE_GREY_500,
            ),
            api_url,
            ft.Row([pick_btn, camera_btn], wrap=True),
            preview_box,
            ft.Row([reset_btn, progress], wrap=True),
            error_text,
            result_card,
        ],
    )

    page.add(ft.SafeArea(content=ft.Column([main_panel, camera_panel])))


if __name__ == "__main__":
    ft.run(main)