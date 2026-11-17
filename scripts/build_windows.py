"""Script đóng gói ứng dụng Flet Desktop trên Windows 11.

Áp dụng runtime patch cho lỗi binary open với encoding trong flet_cli
mà không làm thay đổi trực tiếp file mã nguồn trong thư viện .venv.
"""

from __future__ import annotations

import builtins
import sys
from pathlib import Path

# Runtime Monkeypatch cho builtins.open
# flet_cli.__pyinstaller.win_utils gọi open(exe_path, "ab", encoding="utf-8")
# Trong Python 3.12, mở file mode nhị phân ("b") kèm encoding sẽ văng ValueError.
_orig_open = builtins.open


def _safe_open(*args, **kwargs):
    mode = kwargs.get("mode")
    if mode is None and len(args) > 1:
        mode = args[1]
    if mode and "b" in mode:
        kwargs.pop("encoding", None)
        kwargs.pop("errors", None)
    return _orig_open(*args, **kwargs)


builtins.open = _safe_open

if __name__ == "__main__":
    from flet_cli.cli import main

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    ENTRY_POINT = str(PROJECT_ROOT / "cccd-reader-fe-main" / "main.py")

    sys.argv = [
        "flet",
        "pack",
        ENTRY_POINT,
        "-n",
        "CCCD_Reader",
        "-D",
        "-y",
        "--product-name",
        "CCCD & GPLX Reader",
        "--file-description",
        "Ung dung doc the CCCD va GPLX",
    ]
    print(f"Building Windows Executable with args: {sys.argv}")
    sys.exit(main())
