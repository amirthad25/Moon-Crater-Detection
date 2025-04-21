import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import math
from PIL import Image
import io
import pandas as pd

# Streamlit page configuration
st.set_page_config(page_title="Lunar Crater Detection", page_icon="🌑", layout="centered")

# Title and description
st.title("🌑 Lunar Crater Detection")
st.write("Upload a high-resolution lunar image (e.g., from Chandrayaan-2 OHRC) to detect craters and estimate their rim diameters using YOLOv12.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read the uploaded image
    try:
        # Convert uploaded file to OpenCV format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            st.error("Failed to load the image. Please upload a valid image file.")
        else:
            # Display the uploaded image
            st.image(img, channels="BGR", caption="Uploaded Image", use_container_width=True)

            # Load the trained model
            try:
                model = YOLO("best.pt")
            except Exception as e:
                st.error(f"Failed to load the model: {str(e)}. Please ensure 'best.pt' is in the correct directory.")
                st.stop()

            # Resize image to match YOLO model input size
            input_size = 512
            img_resized = cv2.resize(img, (input_size, input_size))

            # Perform inference
            with st.spinner("Detecting craters..."):
                results = model(img_resized)

            # Set thresholds
            CONFIDENCE_THRESHOLD = 0.8
            NMS_THRESHOLD = 0.4

            # Process results and filter detections
            filtered_boxes = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    confidence = box.conf[0].item()
                    if confidence >= CONFIDENCE_THRESHOLD:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        filtered_boxes.append((x1, y1, x2, y2, confidence))

            # Apply Non-Maximum Suppression (NMS)
            detected_craters = []
            if len(filtered_boxes) > 0:
                indices = cv2.dnn.NMSBoxes(
                    [box[:4] for box in filtered_boxes],
                    [box[4] for box in filtered_boxes],
                    CONFIDENCE_THRESHOLD,
                    NMS_THRESHOLD
                )

                if isinstance(indices, tuple) or len(indices) == 0:
                    st.warning("No craters detected after NMS.")
                else:
                    # Draw filtered detections
                    for idx, i in enumerate(indices):
                        i = i[0] if isinstance(i, (list, tuple)) else i
                        x1, y1, x2, y2, confidence = filtered_boxes[i]

                        # Calculate crater diameter in meters
                        w = x2 - x1
                        h = y2 - y1
                        diameter_meters = math.sqrt(w * h) * 0.25

                        # Store detection details
                        crater_id = idx + 1  # 1-based ID
                        detected_craters.append({
                            "crater_id": crater_id,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "diameter_meters": diameter_meters
                        })

                        # Draw bounding box
                        cv2.rectangle(img_resized, (x1, y1), (x2, y2), (255, 0, 255), 1)

                        # Display the diameter with crater ID
                        label = f"Crater {crater_id}: {diameter_meters:.2f} m"
                        org = (x1, y1 - 10) if y1 - 10 > 10 else (x1, y1 + 20)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 0.5
                        color = (255, 255, 255)
                        thickness = 1
                        cv2.putText(img_resized, label, org, font, fontScale, color, thickness)

            else:
                st.warning("No craters detected!")

            # Display the result image
            st.image(img_resized, channels="BGR", caption="Detected Craters", use_container_width=True)

            # Show detection summary
            if detected_craters:
                st.subheader("Detected Craters")
                crater_data = [
                    {"Crater ID": crater["crater_id"], "Diameter (m)": f"{crater['diameter_meters']:.2f}"}
                    for crater in detected_craters
                ]
                st.table(crater_data)

                # Convert crater data to CSV
                df = pd.DataFrame(crater_data)
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                csv_bytes = csv_buffer.getvalue().encode('utf-8')

                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    # Download result image
                    img_pil = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
                    img_buffer = io.BytesIO()
                    img_pil.save(img_buffer, format="PNG")
                    byte_im = img_buffer.getvalue()
                    st.download_button(
                        label="Download Result Image",
                        data=byte_im,
                        file_name="crater_detection_result.png",
                        mime="image/png"
                    )
                with col2:
                    # Download CSV
                    st.download_button(
                        label="Download Crater Data (CSV)",
                        data=csv_bytes,
                        file_name="crater_data.csv",
                        mime="text/csv"
                    )

            else:
                st.info("No craters to display.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

else:
    st.info("Please upload an image to start crater detection.")

# Footer
st.markdown("---")
st.write("Built for lunar crater analysis using YOLOv12. Pixel scale: 0.25 m/pixel.")