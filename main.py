from HM_Preprocessing import ImageProcessor
from HM_Segmentation import SamSegmenter
from HM_MaskClassifier import MaskClassifier
from HM_OCR import OcrProcessor
from HM_GeospatialConverter import GeospatialConverter
import sys
import cv2 # cv2 is no longer needed here if not used elsewhere

# The visualize_ocr_results helper function is now removed from this file.

def run_full_pipeline():
    """
    Runs the full pipeline with visualization at each step.
    """
    # --- Configuration ---
    CONFIG = {
        'image_path': '/content/drive/MyDrive/HM Registry/stockton_1.png',
        'sam_checkpoint_path': '/content/drive/MyDrive/HM Registry/Segmentation Model/sam_vit_h_4b8939.pth',
        'crop_coords': (445, 637, 3665, 3805),
        'max_dimension': 1500, 'min_red_pixels': 100,
        'max_area': 90000, 'land_vs_text_area_threshold': 4000,
        'gcp_pixel': [[360, 160], [364, 160], [362, 164]],
        'gcp_bng': [[336000, 516000], [336400, 516000], [336200, 516400]],
        'output_filename': 'all_land_parcels.gpkg'
    }

    try:
        # --- Preprocessing, Segmentation, Classification ---
        print("\n--- STEP 1: PREPROCESSING ---")
        processor = ImageProcessor(CONFIG['image_path'])
        processed_image = processor.preprocess(
            crop_coords=CONFIG['crop_coords'], max_dim=CONFIG['max_dimension']
        )
        ImageProcessor.display_image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), title='1. Original Image')
        
        red_mask = processor.isolate_red(processed_image)
        ImageProcessor.display_image(red_mask, title='2. Isolated Red Mask', cmap='gray')

        print("\n--- STEP 2: SEGMENTATION ---")
        segmenter = SamSegmenter(CONFIG['sam_checkpoint_path'])
        raw_sam_masks = segmenter.generate_masks(processed_image)
        segmenter.display_all_masks(processed_image, raw_sam_masks)

        print("\n--- STEP 3: CLASSIFICATION ---")
        classifier = MaskClassifier(
            min_red_pixels=CONFIG['min_red_pixels'], max_area=CONFIG['max_area'],
            area_threshold=CONFIG['land_vs_text_area_threshold']
        )
        classified_masks = classifier.classify_masks(raw_sam_masks, red_mask)
        display_overlays = classifier.create_visualization_overlays(classified_masks, processed_image)
        classifier.display_results(display_overlays, classified_masks)

        # --- 5. OCR Processing ---
        print("\n--- STEP 4: OCR ---")
        text_candidates = classified_masks.get('text', [])
        ocr = OcrProcessor()
        ocr_results = ocr.recognize_text_from_masks(text_candidates, processed_image)
        
        # Visualize by calling the static method directly from the class
        OcrProcessor.visualize_ocr_results(processed_image, ocr_results)
        
        # --- 6. OCR Text Output  ---
        print("\n---  FINAL TEXT RESULTS ---")
        if ocr_results:
            print("Recognized reference numbers and text:")
            for result in ocr_results:
                print(f"- {result['text']}")
        else:
            print("No text was recognized.")

        # --- 7. Georeferencing and File Creation ---
        print("\n--- STEP 5: GEOREFERENCING ---")
        final_text_list = [res['text'] for res in ocr_results]
        geo_converter = GeospatialConverter(
            pixel_points=CONFIG['gcp_pixel'], bng_points=CONFIG['gcp_bng']
        )
        
        # Note: We need the mask arrays, not the full dictionaries, for contour finding
        land_mask_arrays = [mask_info['segmentation'] for mask_info in classified_masks.get('land', [])]
        
        geolocated_polygons = geo_converter.geolocate_masks(land_mask_arrays)
        
        GeospatialConverter.create_geospatial_file(
            polygons=geolocated_polygons,
            recognized_texts=final_text_list,
            filename=CONFIG['output_filename']
        )
        
        print("\n Pipeline finished successfully.")

    except (FileNotFoundError, RuntimeError) as e:
        print(f"A critical error occurred: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)

if __name__ == "__main__":
    run_full_pipeline()