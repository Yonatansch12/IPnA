# -*- coding: utf-8 -*-
"""Streamlit.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1WFi97P24bpPthOecL5gVClLCXmcSC5Vf
"""


import streamlit as st
import cv2
import numpy as np

# Function to detect AruCo marker and calculate real-world scaling factor
def detect_aruco_and_scale(image, marker_length_cm):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        # Assuming we use the first detected marker
        marker_corners = corners[0][0]
        pixel_width = np.linalg.norm(marker_corners[0] - marker_corners[1])
        scale = marker_length_cm / pixel_width  # Scale: cm per pixel
        return scale, corners
    else:
        st.warning("No AruCo marker detected. Please upload a valid image.")
        return None, None

# Function to calculate the real area of an object
def calculate_object_area(image, scale):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        pixel_area = cv2.contourArea(largest_contour)
        real_area_cm2 = (pixel_area * (scale ** 2))
        return real_area_cm2, largest_contour
    else:
        st.warning("No object detected. Please ensure the image includes an insect.")
        return None, None

# Streamlit App
st.title("Real-World Area Calculator Using AruCo Marker")

# User inputs the known length of the AruCo marker
marker_length_cm = st.number_input("Enter the real-world size of the AruCo marker (in cm):", min_value=1.0, value=5.0)

uploaded_file = st.file_uploader("Upload an image containing an AruCo marker and an insect:", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load image
    image = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Detect AruCo and calculate scale
    scale, corners = detect_aruco_and_scale(image, marker_length_cm)

    if scale:
        st.image(cv2.aruco.drawDetectedMarkers(image.copy(), corners), caption="Detected AruCo Marker", channels="BGR")

        # Calculate object (insect) area
        real_area_cm2, contour = calculate_object_area(image, scale)

        if real_area_cm2:
            st.write(f"### Real-world area of the object: **{real_area_cm2:.2f} cm²**")
            # Draw the contour on the image
            result_image = cv2.drawContours(image.copy(), [contour], -1, (0, 255, 0), 2)
            st.image(result_image, caption="Detected Insect Contour", channels="BGR")




