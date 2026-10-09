import streamlit as st
import cv2
import matplotlib.pyplot as plt
import time
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from vision.tracking import (process_motion, initialize_tracker, update_tracker, draw_tracking_box)
from vision.io import (load_image, get_image_info, encode_image, open_webcam)
from vision.preprocessing import (to_grayscale, adjust_brightness, adjust_contrast, gaussian_blur, median_blur, bilateral_filter, calculate_histogram, 
    histogram_equalization, apply_clahe, global_threshold, otsu_threshold, adaptive_threshold)
from vision.detection import (canny_edges, sobel_edges, find_contours, measure_contour, draw_bounding_boxes, count_objects, scan_document)
from vision.matching import feature_match

# PAGE CONFIGURATION
st.set_page_config(page_title="Computer Vision Workstation", page_icon="👁️", layout="wide")

# HEADER
st.title("Computer Vision Workstation")
st.write(
    "A practical Computer Vision application built with "
    "Python, OpenCV, NumPy and Streamlit."
)

# SIDEBAR NAVIGATION
st.sidebar.title("Navigation")
module = st.sidebar.selectbox(
    "Select Module",
    [
        "Image Inspector",
        "Preprocessing",
        "Object Detection & Counting",
        "Document Scanner",
        "Motion Detection",
        "Object Tracking",
        "Feature Matching",
    ],
)

# IMAGE INSPECTOR
if module == "Image Inspector":
    st.header("Image Inspector")
    st.write(
        "Upload an image to inspect its dimensions, "
        "channels and pixel information."
    )
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded_file is not None:
        try:
            # Read uploaded file
            image_bytes = uploaded_file.getvalue()
            # Convert bytes → OpenCV image
            image = load_image(image_bytes)

            # Get image information
            info = get_image_info(image)

            # Convert BGR → RGB for Streamlit
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # DISPLAY IMAGE
            st.subheader("Image")
            st.image(image_rgb, caption=uploaded_file.name, use_container_width=True)

            # IMAGE INFORMATION
            st.subheader("Image Information")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Width", f"{info['width']} px")
            with col2:
                st.metric("Height", f"{info['height']} px")
            with col3:
                st.metric("Channels", info["channels"])
            with col4:
                st.metric("Data Type", info["dtype"])
            st.write("Shape:", info["shape"])

            # EXPORT
            st.subheader("Export")
            png_bytes = encode_image(image, ".png")
            st.download_button(label="Download Image", data=png_bytes, file_name="processed_image.png", mime="image/png")
        except Exception as error:
            st.error(f"Error: {error}")

# OTHER MODULES
elif module == "Preprocessing":
    st.header("Image Enhancement & Preprocessing")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="preprocessing_upload")
    if uploaded_file is not None:
        try:
            image = load_image(uploaded_file.getvalue() )

            # SELECT OPERATION
            operation = st.selectbox(
                "Select Processing Operation",
                [
                    "Grayscale",
                    "Brightness",
                    "Contrast",
                    "Gaussian Blur",
                    "Median Blur",
                    "Bilateral Filter",
                    "Histogram Equalization",
                    "CLAHE",
                    "Global Threshold",
                    "Otsu Threshold",
                    "Adaptive Threshold",
                ],
)
            # PARAMETERS + PROCESSING
            if operation == "Grayscale":
                processed = to_grayscale(image)
            elif operation == "Brightness":
                value = st.slider( "Brightness", -255, 255, 0)
                processed = adjust_brightness(image, value)
            elif operation == "Contrast":
                value = st.slider("Contrast", 0.1, 3.0, 1.0, 0.1)
                processed = adjust_contrast(image, value)
            elif operation == "Gaussian Blur":
                kernel = st.slider("Kernel Size", 1, 15, 5, 2)
                processed = gaussian_blur(image, kernel,)
            elif operation == "Median Blur":
                kernel = st.slider("Kernel Size", 1, 15, 5, 2)
                processed = median_blur(image, kernel)
            elif operation == "Bilateral Filter":
                diameter = st.slider("Diameter", 1, 15, 9, 2)
                processed = bilateral_filter(image, diameter)
            elif operation == "Histogram Equalization":
                processed = histogram_equalization(image)
            elif operation == "CLAHE":
                clip_limit = st.slider("Clip Limit", 1.0, 10.0, 2.0, 0.5)
                processed = apply_clahe(image, clip_limit)
            elif operation == "Global Threshold":
                threshold = st.slider("Threshold", 0, 255, 127 )
                processed = global_threshold(image, threshold)
            elif operation == "Otsu Threshold":
                processed = otsu_threshold(image)
            elif operation == "Adaptive Threshold":
                block_size = st.slider("Block Size", 3, 31, 11, 2)
                constant = st.slider("Constant", 0, 20, 2)
                processed = adaptive_threshold(image, block_size, constant)
                
            # DISPLAY RESULTS
            st.subheader("Result")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Original")
                original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                st.image(original_rgb, use_container_width=True)
            with col2:
                st.write("Processed")
                if len(processed.shape) == 2:
                    st.image(processed, use_container_width=True)
                else:
                    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                    st.image(processed_rgb, use_container_width=True)

            # HISTOGRAM
            st.subheader("Image Histogram")
            histogram = calculate_histogram(processed)
            fig, ax = plt.subplots()
            ax.plot(histogram)
            ax.set_title("Grayscale Histogram")
            ax.set_xlabel("Pixel Intensity")
            ax.set_ylabel("Frequency")
            ax.set_xlim([0, 256])
            st.pyplot(fig)
            plt.close(fig)    
            # DOWNLOAD
            output_bytes = encode_image(processed,".png" )
            st.download_button("Download Result", data=output_bytes, file_name="processed_image.png", mime="image/png", )
        except Exception as error:
            st.error(f"Processing error: {error}")
elif module == "Object Detection & Counting":
    st.header("Object Detection & Counting")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="detection_upload")
    if uploaded_file is not None:
        image = load_image(uploaded_file.getvalue())
        st.subheader("Detection Method")
        method = st.selectbox("Select detection method",
            [
                "Canny Edge Detection",
                "Sobel Edge Detection",
                "Contour Analysis",
                "Object Counting",
            ]
        )
        if method == "Canny Edge Detection":
            low_threshold = st.slider("Low Threshold", min_value=0, max_value=255, value=100)
            high_threshold = st.slider("High Threshold", min_value=0, max_value=255, value=200 )
            result = canny_edges(image, low_threshold, high_threshold)
        elif method == "Sobel Edge Detection":
            kernel_size = st.select_slider("Sobel Kernel Size", options=[1, 3, 5, 7], value=3 )
            result = sobel_edges(image, kernel_size)
        elif method == "Contour Analysis":
            threshold = st.slider("Threshold", min_value=0, max_value=255, value=127)
            min_area = st.number_input("Minimum Object Area", min_value=0.0, value=100.0)
            contours = find_contours(image, threshold)
            result = draw_bounding_boxes(image, contours, min_area )
            valid_contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area ]
            st.write(f"Detected objects: {len(valid_contours)}")
        else:
            threshold = st.slider("Threshold", min_value=0, max_value=255, value=127)
            min_area = st.number_input("Minimum Object Area", min_value=0.0, value=100.0)
            contours = find_contours(image, threshold)
            object_count = count_objects(contours, min_area)
            result = draw_bounding_boxes(image, contours, min_area)
            st.metric("Objects Detected", object_count)
        st.subheader("Result")
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)
        with col2:
            if len(result.shape) == 2:
                st.image(result, caption="Processed Result", use_container_width=True)
            else:
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Processed Result", use_container_width=True)
        st.download_button("Download Result", data=encode_image(result), file_name="detection_result.png", mime="image/png")
elif module == "Document Scanner":
    st.header("Document Scanner")
    uploaded_file = st.file_uploader("Upload a document image", type=["jpg", "jpeg", "png"], key="scanner_upload")
    if uploaded_file is not None:
        image = load_image(uploaded_file.getvalue())
        try:
            edges, contour, scanned = scan_document(image)
            # Draw detected document boundary
            boundary_image = image.copy()
            cv2.drawContours(boundary_image, [contour], -1, (0, 255, 0), 3)
            st.subheader("Document Boundary")
            st.image(
                cv2.cvtColor(boundary_image, cv2.COLOR_BGR2RGB), caption="Detected Document", use_container_width=True)
            st.subheader("Scanned Document")
            st.image(
                cv2.cvtColor(scanned, cv2.COLOR_BGR2RGB), caption="Perspective Corrected", use_container_width=True)
            st.download_button("Download Scanned Document", data=encode_image(scanned), file_name="scanned_document.png", mime="image/png")
        except ValueError as error:
            st.error(str(error))
elif module == "Motion Detection":
    st.header("Motion Detection")
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"], key="motion_video")
    if uploaded_file is not None:
        # Save uploaded video temporarily
        video_path = "output/example.webm"
        with open(video_path, "wb") as file:
            file.write(uploaded_file.getbuffer())
        threshold = st.slider("Motion Threshold", min_value=1, max_value=100, value=25)
        min_area = st.number_input("Minimum Motion Area", min_value=50, value=500, step=50)
        start = st.button("Start Motion Detection")
        if start:
            capture = cv2.VideoCapture(video_path)
            if not capture.isOpened():
                st.error("Unable to open the video.")
            else:
                frame_placeholder = st.empty()
                status_placeholder = st.empty()
                fps_placeholder = st.empty()
                previous_frame = None
                frame_count = 0
                start_time = time.time()
                while True:
                    success, frame = capture.read()
                    if not success:
                        break
                    if previous_frame is None:
                        previous_frame = frame.copy()
                        continue
                    (motion_mask, _, regions, result) = process_motion(previous_frame, frame, threshold, min_area)
                    previous_frame = frame.copy()
                    frame_count += 1
                    elapsed = (time.time() - start_time)
                    fps = (frame_count / elapsed
                        if elapsed > 0 else 0)
                    if regions:
                        status_placeholder.success(f"Motion detected: "
                            f"{len(regions)} region(s)")
                    else:
                        status_placeholder.info("No motion detected")
                    fps_placeholder.write(f"FPS: {fps:.2f}")
                    frame_placeholder.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB ),
                        channels="RGB",
                        use_container_width=True)
                capture.release()
                st.success( "Motion detection completed.")
elif module == "Object Tracking":
    st.header("Object Tracking")
    st.write(
        "Select a video source, choose an object, "
        "then track it across the frames."
    )

    # INPUT SOURCE
    source = st.radio("Select Input Source", ["Upload Video", "Webcam"], horizontal=True)
    tracker_type = st.selectbox("Select Tracker", ["CSRT", "KCF"] )
    # UPLOAD VIDEO
    if source == "Upload Video":
        uploaded_file = st.file_uploader(
            "Upload a video", type=["mp4", "avi", "mov", "mkv"], key="tracking_video")
        if uploaded_file is not None:
            video_path = "output/example.webm"
            with open(video_path, "wb") as file:
                file.write(uploaded_file.getbuffer())
            capture = cv2.VideoCapture(video_path)
            if not capture.isOpened():
                st.error("Unable to open video.")
            else:
                success, first_frame = capture.read()
                if not success:
                    capture.release()
                    st.error("Unable to read the first frame.")
                else:
                    # Convert for Streamlit display
                    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
                    st.subheader("Select Object to Track")
                    st.write(
                        "Draw a rectangle around the "
                        "object you want to track."
                    )
                    background_image = Image.fromarray(first_frame_rgb)
                    canvas_result = st_canvas(
                        fill_color="rgba(0, 255, 0, 0.15)",
                        stroke_width=3,
                        stroke_color="#00FF00",
                        background_image=background_image,
                        update_streamlit=True,
                        height=first_frame.shape[0],
                        width=first_frame.shape[1],
                        drawing_mode="rect",
                        key="tracking_canvas"
                    )
                    start_tracking = st.button("Start Tracking")
                    if start_tracking:
                        if (canvas_result.json_data is None or not canvas_result.json_data["objects"]):
                            st.error("Draw a rectangle around the object first.")
                        else:
                            selected_object = (canvas_result.json_data["objects"][-1])
                            x = int(selected_object["left"])
                            y = int(selected_object["top"])
                            width = int(selected_object["width"] * selected_object.get("scaleX", 1))
                            height = int( selected_object["height"] * selected_object.get( "scaleY", 1))
                            bbox = (x, y, width, height)
                            try:
                                tracker = (initialize_tracker(first_frame, bbox, tracker_type))
                                st.success("Tracker initialized.")
                                frame_placeholder = (st.empty())
                                status_placeholder = (st.empty())
                                fps_placeholder = (st.empty())
                                frame_count = 0
                                start_time = time.time()
                                # Show first frame
                                initial_frame = (draw_tracking_box( first_frame, bbox, True))
                                frame_placeholder.image(cv2.cvtColor(initial_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                                while True:
                                    success, frame = (capture.read())
                                    if not success:
                                        break
                                    tracking_success, new_bbox = (update_tracker( tracker, frame))
                                    result = (draw_tracking_box(frame, new_bbox, tracking_success))
                                    frame_count += 1
                                    elapsed = (time.time() - start_time)
                                    fps = ( frame_count / elapsed if elapsed > 0 else 0 )
                                    frame_placeholder.image(cv2.cvtColor( result, cv2.COLOR_BGR2RGB), channels="RGB",
                                        use_container_width=True
                                    )
                                    fps_placeholder.write(f"FPS: {fps:.2f}" )
                                    if tracking_success:
                                        status_placeholder.success( "Tracking active")
                                    else:
                                        status_placeholder.error("Tracking lost")
                                        break
                                capture.release()
                                st.success("Tracking completed.")
                            except ( ValueError, RuntimeError) as error:
                                capture.release()
                                st.error(str(error))

    # WEBCAM
    else:
        st.subheader("Webcam Object Tracking")
        st.write(
            "Capture a webcam frame, select the object, "
            "then start live tracking.")
        # CAPTURE FIRST FRAME
        if st.button("Capture Webcam Frame"):
            try:
                camera = open_webcam()
                success, frame = camera.read()
                camera.release()
                if not success:
                    st.error("Unable to capture webcam frame.")
                else:
                    st.session_state["tracking_camera_frame"] = frame
                    st.rerun()
            except ValueError as error:
                st.error(str(error))
        # SHOW CAPTURED FRAME
        if "tracking_camera_frame" in st.session_state:
            first_frame = st.session_state["tracking_camera_frame" ]
            first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            st.subheader("Select Object to Track")
            st.write("Draw a rectangle around the object." )
            background_image = Image.fromarray(first_frame_rgb)
            canvas_result = st_canvas(
                fill_color="rgba(0, 255, 0, 0.15)",
                stroke_width=2,
                stroke_color="#00FF00",
                background_image=background_image,
                update_streamlit=True,
                height=first_frame.shape[0],
                width=first_frame.shape[1],
                drawing_mode="rect",
                key="webcam_tracking_canvas"
            )
            start_webcam = st.button("Start Webcam Tracking")
            if start_webcam:
                if (
                    canvas_result.json_data is None
                    or not canvas_result.json_data["objects"]
                ):
                    st.error("Draw a rectangle around the object first.")
                else:
                    selected_object = (canvas_result.json_data["objects"][-1])
                    x = int(selected_object["left"])
                    y = int(selected_object["top"])
                    width = int(selected_object["width"] * selected_object.get("scaleX", 1))
                    height = int(selected_object["height"] * selected_object.get("scaleY", 1))
                    bbox = (x, y, width, height)
                    try:
                        camera = open_webcam()
                        success, current_frame = (camera.read())
                        if not success:
                            camera.release()
                            st.error("Unable to read from webcam.")
                        else:
                            tracker = ( initialize_tracker(current_frame, bbox, tracker_type))
                            frame_placeholder = (st.empty())
                            status_placeholder = (st.empty())
                            fps_placeholder = (st.empty() )
                            frame_count = 0
                            start_time = time.time()
                            while True:
                                success, frame = (camera.read())
                                if not success:
                                    break
                                tracking_success, new_bbox = (update_tracker(tracker,frame))
                                result = (draw_tracking_box(frame, new_bbox, tracking_success))
                                frame_count += 1
                                elapsed = ( time.time() - start_time)
                                fps = (frame_count / elapsed if elapsed > 0 else 0)
                                frame_placeholder.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                                    channels="RGB",
                                    use_container_width=True
                                )
                                fps_placeholder.write(f"FPS: {fps:.2f}")
                                if tracking_success:
                                    status_placeholder.success("Tracking active")
                                else:
                                    status_placeholder.error("Tracking lost")
                                    break
                            camera.release()
                    except (ValueError, RuntimeError) as error:
                        st.error(str(error))
elif module == "Feature Matching":
    st.header("Feature Matching")
    st.write(
        "Compare two images using ORB keypoints, "
        "descriptors, and feature matching."
    )
    # INPUT IMAGES
    col1, col2 = st.columns(2)
    with col1:
        image1_file = st.file_uploader( "Upload Reference Image", type=["jpg", "jpeg", "png"], key="matching_image_1" )
    with col2:
        image2_file = st.file_uploader("Upload Comparison Image", type=["jpg", "jpeg", "png"], key="matching_image_2" )
    # SETTINGS
    n_features = st.slider( "Number of ORB Features", min_value=100, max_value=3000, value=1000, step=100)
    max_matches = st.slider( "Maximum Matches", min_value=10, max_value=100, value=50, step=10)
    # PROCESS
    if image1_file is not None and image2_file is not None:
        image1 = load_image( image1_file.getvalue())
        image2 = load_image(image2_file.getvalue())
        st.subheader("Input Images")
        col1, col2 = st.columns(2)
        with col1:
            st.image(
                cv2.cvtColor( image1, cv2.COLOR_BGR2RGB),
                caption="Reference Image", use_container_width=True)
        with col2:
            st.image(
                cv2.cvtColor(image2, cv2.COLOR_BGR2RGB),
                caption="Comparison Image", use_container_width=True )
        if st.button("Find Feature Matches"):
            try:
                result = feature_match( image1, image2, n_features=n_features, max_matches=max_matches)
                # KEYPOINT INFORMATION
                keypoints1 = result["keypoints1"]
                keypoints2 = result["keypoints2"]
                matches = result["matches"]
                quality = result["quality" ]
                st.subheader("Feature Detection")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric( "Reference Keypoints", len(keypoints1))
                with col2:
                    st.metric("Comparison Keypoints", len(keypoints2))
                with col3:
                    st.metric( "Feature Matches", quality["match_count"])
                # MATCH VISUALIZATION
                st.subheader("ORB Feature Matches")
                visualization = result["visualization"]
                st.image(cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB),
                    caption="ORB Feature Matches", use_container_width=True)
                # MATCH QUALITY
                st.subheader("Matching Statistics")
                col1, col2 = st.columns(2)
                with col1:
                    if quality["average_distance"] is not None:
                        st.metric("Average Hamming Distance", f"{quality['average_distance']:.2f}" )
                with col2:
                    st.metric("Good Matches", quality["good_matches"] )
                # EXPORT
                output_bytes = encode_image(visualization, ".png")
                st.download_button("Download Match Result", data=output_bytes, file_name="feature_matches.png", mime="image/png")
            except ValueError as error:
                st.error(str(error))
    else:
        st.info(
            "Upload two images to perform "
            "feature matching."
            )