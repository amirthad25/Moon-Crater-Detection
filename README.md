Tittle: Moon Crater Rim Size Prediction Using Yolo V12 🏆
colorFrom: pink
colorTo: blue
sdk: streamlit
sdk_version: 1.44.1
app_file: app.py
pinned: false
license: mit
short_description: 'Uses YOLOv12 to detect lunar craters and estimate diameters '

🛠️ Technologies Used
Streamlit: For the interactive and user-friendly web interface.
YOLOv12 (Ultralytics): For high-performance object detection.
OpenCV: For image processing, drawing bounding boxes, and transformations.
NumPy & Pandas: For data manipulation and result handling.
Pillow (PIL): For image encoding and exporting.
CSV Exporting: To download detection results for further analysis.

📌 Features
Image Uploading: Accepts .png, .jpg, or .jpeg high-resolution lunar images.
Crater Detection: Automatically identifies craters using YOLOv12 and filters predictions by confidence and non-maximum suppression (NMS).
Diameter Estimation: Calculates the approximate diameter of each detected crater in meters using a predefined pixel-to-meter scale (0.25 m/pixel).
Visualization: Annotates each crater with bounding boxes and size estimates directly on the image.
Data Table: Displays a summary of detected crater IDs and their measured diameters.

Download Options:
📷 Annotated Image as PNG
📄 Crater Data as CSV

⚙️ How It Works
User Uploads an Image: The image is read and resized to fit the YOLO input dimensions.
Detection is Performed: The YOLOv12 model identifies potential crater regions based on training data.
Filtering & NMS: Only high-confidence predictions are retained and duplicate boxes are removed.
Diameter Calculation: The width and height of the bounding box are used to estimate the crater diameter.
Display & Export: Annotated images and crater details are shown and can be downloaded.

📏 Assumptions
The input imagery has a pixel scale of 0.25 meters/pixel.
The YOLOv12 model (best.pt) has been trained on a dataset of lunar craters.
Input images are relatively clear and captured from lunar orbiter cameras.

🔬 Applications
Astronomy and Planetary Research
Lunar Mapping Projects
Scientific Outreach and Education
AI-based Space Image Analysis

🚀 Future Enhancements
Integration with satellite APIs (e.g., ISRO or NASA) for real-time image feeds.
Improved diameter estimation with elliptical fits or DEM-based depth analysis.
Add confidence threshold sliders for interactive fine-tuning.
Allow multiple image uploads or batch processing.



Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
