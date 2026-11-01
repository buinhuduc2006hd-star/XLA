from pathlib import Path

import cv2


def preprocess_image(input_path: Path, output_path: Path) -> None:
    image = cv2.imread(str(input_path))

    if image is None:
        raise ValueError(f"Cannot read image: {input_path}")

    # 1. Chuyển sang ảnh xám
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Phóng to ảnh 2 lần
    enlarged = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC,
    )

    # 3. Tăng tương phản cục bộ
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced = clahe.apply(enlarged)

    # 4. Làm nét nhẹ
    blurred = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        1.0,
    )

    sharpened = cv2.addWeighted(
        enhanced,
        1.5,
        blurred,
        -0.5,
        0,
    )

    # 5. Lưu ảnh đã xử lý
    success = cv2.imwrite(
        str(output_path),
        sharpened,
    )

    if not success:
        raise ValueError(
            f"Cannot save processed image: {output_path}"
        )
