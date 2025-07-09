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

        Args:
            model_path (str): The path to the SAM checkpoint file.
            model_type (str): The type of SAM model (e.g., 'vit_h').
            device (str, optional): The device to run on ('cuda' or 'cpu'). Defaults to auto-detect.
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

        Args:
            image_bgr (np.ndarray): The input image in BGR format.

        Returns:
            list: A list of dictionaries, where each dict contains a mask and metadata.
        """
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            print("CUDA cache cleared before mask generation.")
            
        print("Generating masks with SAM...")
        # SAM expects an RGB image
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        sam_masks_raw = self.mask_generator.generate(image_rgb)
        print(f"SAM generated {len(sam_masks_raw)} raw masks.")
        return sam_masks_raw

    @staticmethod
    def create_masks_visualization(image_bgr, raw_masks):
        """
        Creates a visual overlay and a combined binary mask from raw SAM output.

        Args:
            image_bgr (np.ndarray): The original BGR image for visualization.
            raw_masks (list): The raw output from the SAM mask generator.

        Returns:
            tuple: A tuple containing (overlay_image, binary_mask_combined).
        """
        if not raw_masks:
            return image_bgr.copy(), np.zeros(image_bgr.shape[:2], dtype=np.uint8)

        overlay = image_bgr.copy()
        binary_mask_combined = np.zeros(image_bgr.shape[:2], dtype=np.uint8)

        for mask_info in raw_masks:
            mask = mask_info['segmentation']
            # Create a random color for the overlay
            color = np.random.randint(0, 255, (3,), dtype=np.uint8)
            
            # Create a colored layer and blend it
            colored_mask_layer = np.zeros_like(overlay)
            colored_mask_layer[mask] = color
            overlay = cv2.addWeighted(overlay, 1, colored_mask_layer, 0.5, 0)
            
            # Add to the combined binary mask
            binary_mask_combined[mask] = 255
            
        return overlay, binary_mask_combined