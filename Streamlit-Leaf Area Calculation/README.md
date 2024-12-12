# Leaf Area Calculator Using AruCo Marker


This Streamlit app calculates the real-world area of a leaf in an uploaded image using an AruCo marker as a reference for scale. It leverages computer vision techniques for accurate and user-friendly measurements.

> [!IMPORTANT]
> The image must contain AruCo 5x5 marker

## Features
Image Upload: Upload an image containing both an AruCo marker and a leaf.

AruCo Marker Detection: Automatically detects the marker and calculates the scaling factor (cm/pixel).

Leaf Detection: Identifies the leaf using contour detection.

Real-World Area Calculation: Computes and displays the leaf's area in square centimeters.
