# import cv2
# import numpy as np
# import easyocr
# from matplotlib import pyplot as plt

# def convert_to_clean_black_white(image_path, show_steps=False):
#     """Enhanced preprocessing with diagnostics"""
#     img = cv2.imread(image_path)
#     if img is None:
#         raise FileNotFoundError(f"Image not found at {image_path}")

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Contrast Limited Adaptive Histogram Equalization (CLAHE)
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#     enhanced = clahe.apply(gray)

#     # Adaptive threshold
#     bw = cv2.adaptiveThreshold(enhanced, 255, 
#                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                              cv2.THRESH_BINARY_INV, 15, 5)

#     # Noise removal (gentler approach)
#     kernel = np.ones((1,1), np.uint8)
#     cleaned = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

#     final = cv2.bitwise_not(cleaned)

#     if show_steps:
#         plt.figure(figsize=(12,6))
#         plt.subplot(121), plt.imshow(gray, cmap='gray'), plt.title('Original Gray')
#         plt.subplot(122), plt.imshow(final, cmap='gray'), plt.title('Processed BW')
#         plt.show()

#     return final

# # 1. Preprocess with visualization
# bw_image = convert_to_clean_black_white('images/1.png', show_steps=True)
# cv2.imwrite('images/processed_bw.png', bw_image)

# # 2. OCR with fallback options
# reader = easyocr.Reader(['en'])

# # First try: Strict mode
# result = reader.readtext(bw_image,
#                        paragraph=False,
#                        width_ths=0.5,
#                        text_threshold=0.6,
#                        allowlist='FBI',
#                        detail=1)  # Keep details for diagnostics

# if not result:
#     print("Strict mode failed. Trying relaxed settings...")
#     # Fallback: Without allowlist
#     result = reader.readtext(bw_image,
#                            paragraph=True,
#                            text_threshold=0.4,
#                            detail=1)

#     # Second fallback: Try Tesseract
#     if not result:
#         import pytesseract
#         text = pytesseract.image_to_string(bw_image, config='--psm 8')
#         print(f"Tesseract fallback: {text.strip()}")
#     else:
#         print(f"Relaxed OCR: {result}")
# else:
#     detected_text = ''.join([res[1] for res in result])
#     confidences = [res[2] for res in result]
#     print(f"Detected: {detected_text} (Confidences: {confidences})")


# import easyocr
# import cv2
# import numpy as np

# # Initialize EasyOCR reader
# reader = easyocr.Reader(['en', 'uz', 'fr'])

# # 1. Read the image
# image = cv2.imread('images/1.png')  

# # 2. Convert to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # 3. Apply adaptive thresholding (better than simple thresholding)
# bw_image = cv2.adaptiveThreshold(
#     gray, 
#     255, 
#     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#     cv2.THRESH_BINARY, 
#     11, 
#     2
# )

# # (Optional) Apply slight blur to reduce noise
# bw_image = cv2.medianBlur(bw_image, 3)

# # (Optional) Save the preprocessed image to check
# cv2.imwrite('images/3_processed.png', bw_image)

# # 4. Perform OCR on the black & white image
# result = reader.readtext(
#     bw_image,
#     paragraph=True,
#     detail=0,
#     decoder='beamsearch',
#     beamWidth=15,  # Increased for better character recognition
#     width_ths=0.5,  # More lenient with character spacing
#     text_threshold=0.6  # Slightly lower to catch faint text
# )

# print(result)


import cv2
import numpy as np
import keras_ocr

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def convert_to_high_contrast_bw(image_path):
    """Convert image to optimized black-and-white for OCR"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast with CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Create sharp black-and-white image
    _, bw = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Invert if needed (black text on white background)
    if np.mean(bw) > 127:
        bw = cv2.bitwise_not(bw)
    
    # Convert to 3-channel image (keras-ocr needs RGB)
    bw_rgb = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)
    
    return bw_rgb

def perform_ocr_kerasocr(image):
    """Run Keras-OCR on preprocessed image"""
    # Pipeline includes detection + recognition
    pipeline = keras_ocr.pipeline.Pipeline()

    # Keras-OCR expects a list of images
    predictions = pipeline.recognize([image])[0]

    # predictions = list of (text, box)
    text_results = [text for text, _ in predictions]
    return ' '.join(text_results) if text_results else ""

# Example usage
if __name__ == "__main__":
    input_image = "images/3.png"

    try:
        bw_image_rgb = convert_to_high_contrast_bw(input_image)
        cv2.imwrite("processed_bw_rgb.png", cv2.cvtColor(bw_image_rgb, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        exit()

    detected_text = perform_ocr_kerasocr(bw_image_rgb)
    print(f"Detected text: {detected_text}")


# import easyocr

# reader = easyocr.Reader(['en', 'uz'])

# # Read the processed image
# result = reader.readtext('processed_bw.png', 
#                        paragraph=True,
#                        detail=0,
#                        decoder='beamsearch',
#                        beamWidth=10,
#                        width_ths=0.3,
#                        text_threshold=0.7)

# print(result)