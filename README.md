1. Terminal 1 — Chạy Backend (EasyOCR)
Mở Terminal trong VS Code tại thư mục gốc Nhận diện giấy tờ.
##cd .\cccd-reader-api-main
##.\run_api.bat
2. Terminal 2 — Chạy Frontend (Flet)
Mở thêm Terminal mới trong VS Code bằng dấu +.

Nếu đang ở thư mục gốc Nhận diện giấy tờ:

##cd .\cccd-reader-fe-main

Vì cậu đã gặp lỗi SSL khi Flet khởi tạo lần đầu, có thể tiếp tục sử dụng lệnh có truststore:

##.\.venv\Scripts\python.exe -c "import truststore; truststore.inject_into_ssl(); from flet_cli.cli import main; main()" run main.py

Hoặc, sau khi Flet Desktop Client đã được tải và lưu vào máy, thử lệnh ngắn hơn:

.\.venv\Scripts\flet.exe run main.py
####
cd "C:\Users\bui nhu duc\XLA\cccd-reader-api-main"

..\.venv\Scripts\python.exe .\scripts\debug_roi.py ".\test_images\cccd_01.jpg" --card-type cccd

start .\debug_output\annotated_cccd_rois.png
####
