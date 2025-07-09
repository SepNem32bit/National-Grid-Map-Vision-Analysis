import easyocr
import cv2

class OcrProcessor:
    """
    A class to perform OCR on image regions using EasyOCR.
    """
    def __init__(self, lang=['en']):
        """
        Initializes the EasyOCR reader. The model is downloaded on first use.

        Args:
            lang (list): A list of language codes for OCR (e.g., ['en']).
        """
        self.reader = easyocr.Reader(lang)
        print(f"\n--- OCR Processor Initialized for language(s): {lang} ---")

    def recognize_text_from_masks(self, text_masks_info, image_bgr, padding=5):
        """
        Extracts text from image regions defined by a list of text masks.

        Args:
            text_masks_info (list): A list of mask_info dictionaries for text candidates.
            image_bgr (np.ndarray): The source image from which to crop text.
            padding (int): Padding to add around the bounding box before OCR.

        Returns:
            list: A list of recognized text strings.
        """
        if not text_masks_info:
            print("No text candidates to perform OCR on.")
            return []
            
        print(f"Performing OCR on {len(text_masks_info)} text candidates...")
        recognized_texts = []
        img_h, img_w = image_bgr.shape[:2]

        for mask_info in text_masks_info:
            # Get the bounding box from the mask info
            # The 'bbox' is in [x, y, width, height] format
            x, y, w, h = mask_info['bbox']
            
            # Crop the original image to this bounding box, applying padding safely
            y_start = max(0, y - padding)
            y_end = min(img_h, y + h + padding)
            x_start = max(0, x - padding)
            x_end = min(img_w, x + w + padding)
            
            cropped_text_image = image_bgr[y_start:y_end, x_start:x_end]

            # Use EasyOCR to read the text from the cropped image
            # detail=0 returns a simple list of strings
            if cropped_text_image.size > 0:
                result = self.reader.readtext(cropped_text_image, detail=0, paragraph=False)
                
                if result:
                    recognized_texts.extend(result)
        
        return recognized_texts