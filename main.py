from HM_Preprocessing import ImageProcessor
from HM_Segmentation import SamSegmenter
from HM_MaskClassifier import MaskClassifier 
from HM_OCR import OcrProcessor  
import sys

def run_full_pipeline():
    """
    Runs the full pipeline: Preprocess -> Segment -> Classify -> OCR -> Report.
    """
    # --- 1. Configuration ---
    CONFIG = {
        'image_path': '/content/drive/MyDrive/HM Registry/stockton_1.png',
        'sam_checkpoint_path': '/content/drive/MyDrive/HM Registry/Segmentation Model/sam_vit_h_4b8939.pth',
        'crop_coords': (445, 637, 3665, 3805),
        'max_dimension': 1500,
        'min_red_pixels': 100,
        'max_area': 90000,
        'land_vs_text_area_threshold': 4000
    }

    try:
        # --- 2. Preprocessing & Segmentation ---
        processor = ImageProcessor(CONFIG['image_path'])
        processed_image = processor.preprocess(
            crop_coords=CONFIG['crop_coords'],
            max_dim=CONFIG['max_dimension']
        )
        red_mask = processor.isolate_red(processed_image)

        segmenter = SamSegmenter(CONFIG['sam_checkpoint_path'])
        raw_sam_masks = segmenter.generate_masks(processed_image)

        # --- 3. Classification ---
        classifier = MaskClassifier(
            min_red_pixels=CONFIG['min_red_pixels'],
            max_area=CONFIG['max_area'],
            area_threshold=CONFIG['land_vs_text_area_threshold']
        )
        classified_masks = classifier.classify_masks(raw_sam_masks, red_mask)
        
        # (Optional) Display the visual classification results
        display_overlays = classifier.create_visualization_overlays(classified_masks, processed_image)
        classifier.display_results(display_overlays, classified_masks)

        # --- 4. OCR Processing ---
        # Note: The 'text' key now holds a list of dictionaries, as needed.
        text_candidates = classified_masks.get('text', [])
        
        ocr = OcrProcessor()
        final_text = ocr.recognize_text_from_masks(text_candidates, processed_image)
        
        # --- 5. Final Results ---
        print("\n--- OCR Results ---")
        if final_text:
            print("Recognized reference numbers and text:")
            for text in final_text:
                print(f"- {text}")
        else:
            print("No text was recognized.")

    except (FileNotFoundError, RuntimeError) as e:
        print(f"A critical error occurred: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)

if __name__ == "__main__":
    run_full_pipeline()