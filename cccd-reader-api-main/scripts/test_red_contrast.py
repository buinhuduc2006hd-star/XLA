import cv2
import numpy as np
from pathlib import Path

img_path = Path("cccd-reader-api-main/debug_output/stage3_1/06_gplx_1794202637809_499_final_d.jpg")
img = cv2.imread(str(img_path))
h, w = img.shape[:2]

# Crop license number area
crop = img[int(h*0.30):int(h*0.43), int(w*0.50):int(w*0.85)]
gray_std = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
green_ch = crop[:, :, 1]  # Green channel inverted gives max contrast for red ink

diff_std = np.max(gray_std) - np.min(gray_std)
diff_green = np.max(green_ch) - np.min(green_ch)
print(f"Standard grayscale dynamic range: {diff_std}")
print(f"Green channel dynamic range: {diff_green}")

