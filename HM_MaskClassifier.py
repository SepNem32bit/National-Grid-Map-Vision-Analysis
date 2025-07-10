import cv2
import numpy as np
import matplotlib.pyplot as plt

class MaskClassifier:
    """
    A class to filter and classify SAM masks based on color, area, and other properties.
    """
    def __init__(self, min_red_pixels=100, max_area=90000, area_threshold=4000):
        """
        Initializes the classifier with property-based thresholds.
        """
        self.min_red_pixels = min_red_pixels
        self.max_area = max_area
        self.area_threshold = area_threshold
        print("--- Mask Classifier Initialized (Property-Based) ---")
        print(f"Min Red Pixels: {self.min_red_pixels}, Max Area: {self.max_area}, Land/Text Threshold: {self.area_threshold}")

    def classify_masks(self, raw_sam_masks, red_hsv_mask):
        """
        Filters masks and then classifies them into 'land' and 'text'.
        """
        candidates = []
        # --- Filter for viable candidates ---
        print("\n--- Filtering for red candidates based on properties ---")
        for mask_info in raw_sam_masks:
            area = mask_info['area']
            if area >= self.max_area:
                continue

            mask_uint8 = mask_info['segmentation'].astype(np.uint8)
            num_red_pixels = cv2.countNonZero(cv2.bitwise_and(mask_uint8, red_hsv_mask))

            if num_red_pixels > self.min_red_pixels:
                candidates.append(mask_info)
        
        print(f"Found {len(candidates)} candidates after filtering.")

        # --- Classify candidates into 'land' and 'text' ---
        land_masks_info = []
        text_masks_info = []
        for mask_info in candidates:
            if mask_info['area'] > self.area_threshold:
                land_masks_info.append(mask_info)
            else:
                text_masks_info.append(mask_info)
        
        print(f"Classification complete: {len(land_masks_info)} land, {len(text_masks_info)} text.")
        
        return {'land': land_masks_info, 'text': text_masks_info}

    def create_visualization_overlays(self, classified_masks, image_bgr):
        """
        Creates separate display images for land and text masks.
        """
        land_display = image_bgr.copy()
        text_display = image_bgr.copy()

        # Loop through the list of dictionaries and get the 'segmentation' key
        for mask_info in classified_masks.get('land', []):
            mask = mask_info['segmentation'] # Extract the mask array
            overlay = np.zeros_like(land_display)
            overlay[mask] = (0, 100, 255) # Red
            land_display = cv2.addWeighted(land_display, 1, overlay, 0.5, 0)
        
        for mask_info in classified_masks.get('text', []):
            mask = mask_info['segmentation'] # Extract the mask array
            overlay = np.zeros_like(text_display)
            overlay[mask] = (0, 255, 255) # Yellow
            text_display = cv2.addWeighted(text_display, 1, overlay, 0.5, 0)
            
        return {'land': land_display, 'text': text_display}

    def display_results(self, display_images, classified_masks):
        """
        Visualizes the classification results in a 2-panel plot.
        """
        land_count = len(classified_masks.get('land', []))
        text_count = len(classified_masks.get('text', []))

        print("\n--- Visualizing Property-Based Classification Results ---")
        plt.figure(figsize=(20, 10))

        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(display_images['land'], cv2.COLOR_BGR2RGB))
        plt.title(f'Classified Land Boundaries ({land_count})')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(display_images['text'], cv2.COLOR_BGR2RGB))
        plt.title(f'Classified Text Candidates ({text_count})')
        plt.axis('off')

        plt.tight_layout()
        plt.show()