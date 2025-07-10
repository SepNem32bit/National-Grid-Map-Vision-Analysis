import easyocr
import cv2
import numpy as np
import matplotlib.pyplot as plt

class OcrProcessor:
    """
    A class to perform OCR and visualize results.
    """
    def __init__(self, lang=['en']):
        """
        Initializes the EasyOCR reader.
        """
        self.reader = easyocr.Reader(lang)
        print(f"\n--- OCR Processor Initialized for language(s): {lang} ---")

    def recognize_text_from_masks(self, text_masks_info, image_bgr, padding=5):
        """
        Extracts text from image regions and returns structured data.
        """
        if not text_masks_info:
            return []
            
        print(f"Performing OCR on {len(text_masks_info)} text candidates...")
        ocr_results = []
        img_h, img_w = image_bgr.shape[:2]

        for mask_info in text_masks_info:
            x, y, w, h = map(int, mask_info['bbox'])
            
            y_start, y_end = max(0, y - padding), min(img_h, y + h + padding)
            x_start, x_end = max(0, x - padding), min(img_w, x + w + padding)
            
            cropped_text_image = image_bgr[y_start:y_end, x_start:x_end]

            if cropped_text_image.size > 0:
                result = self.reader.readtext(cropped_text_image, detail=0, paragraph=True)
                
                if result:
                    ocr_results.append({'bbox': (x, y, w, h), 'text': " ".join(result)})
        
        return ocr_results

    @staticmethod
    def visualize_ocr_results(image, ocr_results):
        """
        Draws OCR text and bounding boxes on an image for visualization.
        """
        print("\n--- Visualizing Final OCR Results ---")
        display_image = image.copy()
        
        for result in ocr_results:
            bbox = result['bbox']
            text = result['text']
            
            x, y, w, h = bbox
            
            # Draw a green rectangle and put the recognized text above it
            cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display_image, text, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
        # Display the final image with text
        plt.figure(figsize=(15, 15))
        plt.imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
        plt.title('Final OCR Results')
        plt.axis('off')
        plt.show()