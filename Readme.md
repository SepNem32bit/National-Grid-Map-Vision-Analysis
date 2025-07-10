# National Grid Map Information System (HM Registry)

This project is a complete computer vision pipeline designed to digitize highlited area and information of historical land registry maps. It automatically identifies land parcels and reference numbers, extracts the text using OCR, and georeferences the parcel boundaries into a final, usable geospatial file.

---

## Features

- **Image Preprocessing**: Automatically crops and resizes map images for optimal processing.
- **Color Isolation**: Isolates features of interest (e.g., red-lined parcels) using HSV color space thresholding.
- **AI-Powered Segmentation**: Utilizes Meta AI's Segment Anything Model (SAM) to generate high-quality masks for all objects on the map.
- **Mask Classification**: A custom classifier filters the hundreds of raw SAM masks to identify "Land Boundary" and "Text" candidates based on their color and physical properties (size, shape).
- **Optical Character Recognition (OCR)**: Extracts text and reference numbers from the classified text masks using `easyocr`.
- **Georeferencing**: Transforms the pixel coordinates of the land parcel boundaries into real-world British National Grid (BNG) coordinates using Ground Control Points (GCPs).
- **Geospatial File Generation**: Saves the final geolocated land parcels and their associated OCR text into a GeoPackage (`.gpkg`) file for use in GIS software like QGIS or ArcGIS.

---

## Pipeline Workflow

1. **Preprocessing (`HM_Preprocessing`)**: The input map is loaded, cropped, and resized. A binary mask of all red elements is created.  
2. **Segmentation (`HM_Segmentation`)**: The preprocessed image is fed into the Segment Anything Model (SAM) to generate masks for every object.  
3. **Classification (`HM_MaskClassifier`)**: The raw SAM masks are filtered. Only masks that contain a sufficient amount of red are kept. These candidates are then classified as "land" or "text" based on their area.  
4. **OCR (`HM_OCR`)**: The text candidate masks are processed by an OCR engine to read the text within their bounding boxes.  
5. **Georeferencing (`HM_GeospatialConverter`)**: The land parcel masks are converted to polygons. Their pixel vertices are transformed into British National Grid coordinates, and the final result is saved as a GeoPackage file with the OCR text as metadata.  

---

## Installation

### 1. Clone the Repository

Run: `git clone https://github.com/YourUsername/HM_Registry.git && cd HM_Registry`

### 2. Create a Virtual Environment

On Windows:  
`python -m venv venv && .\venv\Scripts\activate`  

On macOS/Linux:  
`python3 -m venv venv && source venv/bin/activate`  

### 3. Install Dependencies

Run: `pip install -r requirements.txt`

### 4. Download the SAM Model Checkpoint

Download the `sam_vit_h_4b8939.pth` checkpoint from [Pre-Trained SAM](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth), and place it in a folder called `Segmentation Model` within the project root:  

`Segmentation Model/sam_vit_h_4b8939.pth`

---

## Usage

### 1. Configure the Pipeline

Open `main.py` and edit the `CONFIG` dictionary. You can set:

- `image_path`: path to your input map image  
- `sam_checkpoint_path`: path to the downloaded SAM model  
- `crop_coords`, `max_dimension`: for preprocessing  
- `gcp_pixel` and `gcp_bng`: Ground Control Points for georeferencing  
- `output_filename`: desired name for the `.gpkg` file  

### 2. Run the Pipeline

Execute: `python main.py`

The script will print updates and show visual plots for each processing step.

### 3. Output

The final GeoPackage file (e.g., `all_land_parcels.gpkg`) will be saved in the project root.

---

## Project Structure
```
HM_Registry/
│
├── main.py                     # The main executable script that runs the entire pipeline
│
├── HM_Preprocessing.py         # ImageProcessor class: handles loading, cropping, resizing, and color isolation
├── HM_Segmentation.py          # SamSegmenter class: handles loading SAM and generating masks
├── HM_MaskClassifier.py        # MaskClassifier class: filters and classifies SAM masks
├── HM_OCR.py                   # OcrProcessor class: runs OCR on text masks
├── HM_GeospatialConverter.py   # GeospatialConverter class: handles georeferencing and saving the final file
│
├── requirements.txt            # A list of all Python dependencies for the project
├── README.md                   # This file
│
└── Segmentation Model/
    └── sam_vit_h_4b8939.pth    # The SAM model checkpoint (must be downloaded)
```