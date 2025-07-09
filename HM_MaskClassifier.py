import cv2
import numpy as np
import matplotlib.pyplot as plt

class MaskClassifier:
    """
    A class to filter and classify SAM masks based on color, area, and other properties.
    This version uses a single-pass filtering approach.
    """
    def __init__(self, min_red_pixels=100, max_area=90000, area_threshold=4000):
        """
        Initializes the classifier with property-based thresholds.

        Args:
            min_red_pixels (int): The minimum number of red pixels a mask must contain.
            max_area (int): The maximum area for a mask to be considered a candidate.
            area_threshold (int): The area value to separate 'land' (larger) from 'text' (smaller).
        """
        self.min_red_pixels = min_red_pixels
        self.max_area = max_area
        self.area_threshold = area_threshold
        print("--- Mask Classifier Initialized (Property-Based) ---")
        print(f"Min Red Pixels: {self.min_red_pixels}, Max Area: {self.max_area}, Land/Text Threshold: {self.area_threshold}")

    def classify_masks(self, raw_sam_masks, red_hsv_mask):
        """
        Filters masks based on redness and area, then classifies them into 'land' and 'text'.

        Args:
            raw_sam_masks (list): The raw output from SamSegmenter.
            red_hsv_mask (np.ndarray): The binary mask highlighting red areas.

        Returns:
            dict: A dictionary with lists of masks for 'land' and 'text'.
        """
        candidates = []
        # --- Step 1: Filter for viable candidates in a single pass ---
        print("\n--- Filtering for red candidates based on properties ---")
        for mask_info in raw_sam_masks:
            area = mask_info['area']
            
            # Condition 1: Check if the mask area is within our desired range
            if area >= self.max_area:
                continue

            # Condition 2: Check for minimum red pixel count
            mask_uint8 = mask_info['segmentation'].astype(np.uint8)
            num_red_pixels = cv2.countNonZero(cv2.bitwise_and(mask_uint8, red_hsv_mask))

            if num_red_pixels > self.min_red_pixels:
                candidates.append(mask_info)
        
        print(f"Found {len(candidates)} candidates after filtering.")

        # --- Step 2: Classify candidates into 'land' and 'text' ---
        land_masks = []
        text_masks = []
        for mask_info in candidates:
            if mask_info['area'] > self.area_threshold:
                land_masks.append(mask_info['segmentation'])
            else:
                text_masks.append(mask_info['segmentation'])
        
        print(f"Classification complete: {len(land_masks)} land, {len(text_masks)} text.")
        
        return {'land': land_masks, 'text': text_masks}

    def create_visualization_overlays(self, classified_masks, image_bgr):
        """
        Creates separate display images for land and text masks.

        Args:
            classified_masks (dict): The dictionary containing 'land' and 'text' masks.
            image_bgr (np.ndarray): The source image for creating visualizations.

        Returns:
            dict: A dictionary containing the 'land' and 'text' display images.
        """
        land_display = image_bgr.copy()
        text_display = image_bgr.copy()

        # Draw land masks (semi-transparent red)
        for mask in classified_masks.get('land', []):
            overlay = np.zeros_like(land_display)
            overlay[mask] = (0, 100, 255) # Red
            land_display = cv2.addWeighted(land_display, 1, overlay, 0.5, 0)
        
        # Draw text masks (semi-transparent yellow)
        for mask in classified_masks.get('text', []):
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

        # Plot 1: Land Boundaries
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(display_images['land'], cv2.COLOR_BGR2RGB))
        plt.title(f'Classified Land Boundaries ({land_count})')
        plt.axis('off')

        # Plot 2: Text Candidates
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(display_images['text'], cv2.COLOR_BGR2RGB))
        plt.title(f'Classified Text Candidates ({text_count})')
        plt.axis('off')

        plt.tight_layout()
        plt.show()