import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon

class GeospatialConverter:
    """
    A class to handle georeferencing of masks and creation of geospatial files.
    """
    def __init__(self, pixel_points, bng_points):
        """
        Initializes the converter and calculates the transformation matrix.

        Args:
            pixel_points (list): A list of [x, y] pixel coordinates.
            bng_points (list): A list of corresponding BNG coordinates.
        """
        # Ensure points are in the correct format for OpenCV
        pixel_pts = np.float32(pixel_points)
        bng_pts = np.float32(bng_points)
        self.transform_matrix = cv2.getAffineTransform(pixel_pts, bng_pts)
        print("\n--- Geospatial Converter Initialized ---")
        print("Transformation matrix calculated successfully.")

    def geolocate_masks(self, land_masks):
        """
        Converts a list of binary masks into geolocated Shapely Polygons.

        Args:
            land_masks (list): A list of land boundary mask arrays.

        Returns:
            list: A list of Shapely Polygon objects with BNG coordinates.
        """
        geolocated_polygons = []
        for land_mask in land_masks:
            # findContours requires a uint8 array
            mask_uint8 = land_mask.astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue

            # Transform the vertices of the largest contour
            pixel_vertices = contours[0].reshape(-1, 1, 2).astype(np.float32)
            georeferenced_vertices = cv2.transform(pixel_vertices, self.transform_matrix)
            
            # Squeeze to remove unnecessary dimensions for Polygon creation
            final_bng_coords = georeferenced_vertices.squeeze()
            
            # A valid polygon needs at least 3 vertices
            if len(final_bng_coords) > 2:
                geolocated_polygons.append(Polygon(final_bng_coords))

        print(f"Successfully geolocated {len(geolocated_polygons)} land boundaries.")
        return geolocated_polygons

    @staticmethod
    def create_geospatial_file(polygons, recognized_texts, filename="land_parcels.gpkg", crs="EPSG:27700"):
        """
        Creates a geospatial file from a list of polygons and associated metadata.
        """
        if not polygons:
            print("No geolocated polygons to save.")
            return

        # Prepare metadata to be assigned to each feature
        metadata_text = ", ".join(recognized_texts)
        num_polygons = len(polygons)
        data = {'reference_numbers': [metadata_text] * num_polygons}
        
        # Create the GeoDataFrame
        gdf = gpd.GeoDataFrame(data, geometry=polygons, crs=crs)
        
        # Save to file
        gdf.to_file(filename, driver="GPKG")
        print(f"\nSuccessfully created geospatial file with {num_polygons} features: {filename}")