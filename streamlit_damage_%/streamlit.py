import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Streamlit app title
st.title("Leaf Damage Analysis")

# Instructions
st.write("Upload an image to analyze leaf area, damage, and refined damage.")

# File uploader
uploaded_file = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

# Known ArUco marker size (cm)
aruco_marker_size_cm = 5.0  # Marker size in cm

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Detect ArUco marker and process the image
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # Load ArUco dictionary and parameters
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        aruco_params = cv2.aruco.DetectorParameters()

        # Detect markers
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
        if ids is None or len(ids) == 0:
            st.warning("No ArUco marker detected in the image.")
        else:
            # Calculate pixel-to-cm scale
            marker_corners = corners[0][0]
            pixel_width = np.linalg.norm(marker_corners[0] - marker_corners[1])
            cm_per_pixel = aruco_marker_size_cm / pixel_width

            # Convert image to HSV
            hsv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

            # Leaf mask
            lower_leaf = np.array([25, 40, 40])
            upper_leaf = np.array([100, 255, 255])
            leaf_mask = cv2.inRange(hsv_image, lower_leaf, upper_leaf)
            total_leaf_pixels = cv2.countNonZero(leaf_mask)
            total_leaf_area_cm2 = total_leaf_pixels * (cm_per_pixel ** 2)

            # Damage mask
            lower_damage = np.array([10, 50, 50])
            upper_damage = np.array([35, 255, 255])
            damage_mask = cv2.inRange(hsv_image, lower_damage, upper_damage)
            total_damage_pixels = cv2.countNonZero(damage_mask)
            total_damage_cm2 = total_damage_pixels * (cm_per_pixel ** 2)

            # Remove veins from damage mask
            lower_veins = np.array([30, 40, 40])
            upper_veins = np.array([60, 255, 255])
            veins_mask = cv2.inRange(hsv_image, lower_veins, upper_veins)
            damage_mask_no_veins = cv2.bitwise_and(damage_mask, cv2.bitwise_not(veins_mask))
            refined_damage_pixels = cv2.countNonZero(damage_mask_no_veins)
            refined_damage_cm2 = refined_damage_pixels * (cm_per_pixel ** 2)

            # Damage percentages
            total_damage_percentage = (total_damage_cm2 / total_leaf_area_cm2) * 100 if total_leaf_area_cm2 > 0 else 0
            refined_damage_percentage = (refined_damage_cm2 / total_leaf_area_cm2) * 100 if total_leaf_area_cm2 > 0 else 0

            # Highlight damage areas
            highlighted_damage = cv2.bitwise_and(image_np, image_np, mask=damage_mask)
            highlighted_refined_damage = cv2.bitwise_and(image_np, image_np, mask=damage_mask_no_veins)

            # Results
            st.subheader("Results")
            st.write(f"Total Leaf Area: {total_leaf_area_cm2:.2f} cm²")
            st.write(f"Total Damaged Area: {total_damage_cm2:.2f} cm² ({total_damage_percentage:.2f}%)")
            st.write(f"Refined Damaged Area: {refined_damage_cm2:.2f} cm² ({refined_damage_percentage:.2f}%)")

            # Display results
            st.subheader("Output Images")
            col1, col2, col3 = st.columns(3)
            col1.image(image, caption="Original Image", use_column_width=True)
            col2.image(highlighted_damage, caption="Overall Damaged Area", use_column_width=True)
            col3.image(highlighted_refined_damage, caption="Refined Damaged Area", use_column_width=True)

    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")
