import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

class SamSegmenter:
    """A class to handle segmentation using the Segment Anything Model (SAM)."""

    def __init__(self, model_path, model_type="vit_h", device=None):
        """
        Initializes the SAM model and mask generator.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"--- Initializing SAM from {model_path} on device: {self.device} ---")
        
        try:
            sam = sam_model_registry[model_type](checkpoint=model_path)
            sam.to(device=self.device)
            self.model = sam
            print("SAM model loaded successfully. ✅")
        except FileNotFoundError:
            raise FileNotFoundError(f"SAM checkpoint not found at '{model_path}'. Please check the path.")
        except Exception as e:
            raise RuntimeError(f"Error loading SAM model: {e}")

        # Configure the mask generator
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.model,
            points_per_side=40,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.92,
            min_mask_region_area=30,
        )

    def generate_masks(self, image_bgr):
        """
        Generates segmentation masks for the entire image.
        """
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            print("CUDA cache cleared before mask generation.")
            
        print("Generating masks with SAM...")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        sam_masks_raw = self.mask_generator.generate(image_rgb)
        print(f"SAM generated {len(sam_masks_raw)} raw masks.")
        return sam_masks_raw

    @staticmethod
    def create_masks_visualization(image_bgr, raw_masks):
        """
        Creates a visual overlay and a combined binary mask from raw SAM output.
        """
        if not raw_masks:
            return image_bgr.copy(), np.zeros(image_bgr.shape[:2], dtype=np.uint8)

        # Create the color overlay image
        overlay = image_bgr.copy()
        # Create a blank image for the combined binary mask
        binary_mask_combined = np.zeros(image_bgr.shape[:2], dtype=np.uint8)

        for mask_info in raw_masks:
            mask = mask_info['segmentation']
            
            # Draw a random color on the overlay
            color = np.random.randint(0, 255, (3,), dtype=np.uint8)
            colored_mask_layer = np.zeros_like(overlay)
            colored_mask_layer[mask] = color
            overlay = cv2.addWeighted(overlay, 1, colored_mask_layer, 0.5, 0)
            
            # Draw white on the binary mask
            binary_mask_combined[mask] = 255
            
        return overlay, binary_mask_combined

    def display_all_masks(self, image_bgr, raw_masks):
        """
        Generates and displays a plot of all raw SAM masks, including the binary version.
        """
        print("\n--- Visualizing All Raw SAM Masks ---")
        # Create both the overlay and the binary mask images
        overlay_image, binary_mask = self.create_masks_visualization(image_bgr, raw_masks)

        # Create a figure with two subplots
        plt.figure(figsize=(18, 9))

        # Plot 1: Color Overlay
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
        plt.title(f'All Raw SAM Masks (Overlay, Total: {len(raw_masks)})')
        plt.axis('off')
        
        # Plot 2: Combined Binary Mask
        plt.subplot(1, 2, 2)
        plt.imshow(binary_mask, cmap='gray')
        plt.title('All Raw SAM Masks (Combined Binary)')
        plt.axis('off')

        plt.tight_layout()
        plt.show()