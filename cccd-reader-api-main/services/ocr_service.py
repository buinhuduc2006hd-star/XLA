import easyocr

reader = easyocr.Reader(["vi", "en"], gpu=False)

def read_text(image_path: str):
    results = reader.readtext(
    image_path,
    decoder="beamsearch",
    beamWidth=10,
    mag_ratio=1.5,
    contrast_ths=0.1,
    adjust_contrast=0.5,
)

    converted_results = []

    for box, text, confidence in results:
        converted_results.append(
            {
                "box": [
                [int(point[0]), int(point[1])]
                for point in box
            ],
            "text": text,
            "confidence": float(confidence),
        }
    )

    return converted_results
