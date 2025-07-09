import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageProcessor:
    """A class to handle image processing tasks for historical maps."""

    def __init__(self, image_path):
        """
        Initializes the processor and loads the image.

        Args:
            image_path (str): The path to the input image.
        """
        self.original_bgr = cv2.imread(image_path)
        if self.original_bgr is None:
            raise FileNotFoundError(f"Error: Image not found at {image_path}.")
        print(f"Successfully loaded image from: {image_path}")

    def preprocess(self, crop_coords, max_dim=1500):
        """
        Crops and resizes the image.

        Args:
            crop_coords (tuple): A tuple of (x1, y1, x2, y2) for cropping.
            max_dim (int): The maximum dimension for resizing.

        Returns:
            numpy.ndarray: The preprocessed BGR image.
        """
        x1, y1, x2, y2 = crop_coords
        h_img, w_img = self.original_bgr.shape[:2]

        # Ensure crop coordinates are within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        cropped_img = self.original_bgr[y1:y2, x1:x2]
        print(f"Cropped image shape: {cropped_img.shape}")

        # Resize if necessary
        h_cropped, w_cropped = cropped_img.shape[:2]
        if max(h_cropped, w_cropped) > max_dim:
            scale_factor = max_dim / max(h_cropped, w_cropped)
            new_w = int(w_cropped * scale_factor)
            new_h = int(h_cropped * scale_factor)
            resized_img = cv2.resize(cropped_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"Resized image for processing from {cropped_img.shape} to {resized_img.shape}")
        else:
            resized_img = cropped_img.copy()

        return resized_img

    def isolate_red(self, image_bgr):
        """
        Isolates red colors in the image using HSV color space.

        Args:
            image_bgr (numpy.ndarray): The input BGR image.

        Returns:
            numpy.ndarray: A binary mask of the red areas.
        """
        img_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        # Define HSV ranges for red
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])

        # Create masks and combine them
        mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        return red_mask

    @staticmethod
    def display_image(image, title='Image', cmap=None):
        """
        Displays an image using matplotlib.

        Args:
            image (numpy.ndarray): The image to display.
            title (str): The title for the plot.
            cmap (str, optional): The colormap to use for grayscale images.
        """
        plt.figure(figsize=(8, 6))
        plt.imshow(image, cmap=cmap)
        plt.title(title)
        plt.axis('off')
        plt.show()