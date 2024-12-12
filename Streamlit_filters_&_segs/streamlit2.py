# -*- coding: utf-8 -*-
"""streamlit2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ltb8xdeNCfZl6b3Wp2Ac0uKOdoaizb0D
"""

import streamlit as st
import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from skimage import color

# --- HELPER FUNCTIONS ---

def apply_gaussian_blur(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_sobel(image, ksize):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    sobel = cv2.magnitude(sobelx, sobely)
    return np.uint8(sobel)

def apply_canny(image, threshold1, threshold2):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, threshold1, threshold2)

def apply_otsu(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh_val = threshold_otsu(gray)
    _, otsu = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    return otsu

def apply_watershed(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    unknown = cv2.subtract(sure_bg, np.uint8(sure_fg))
    markers = cv2.connectedComponents(np.uint8(sure_fg))[1]
    markers += 1
    markers[unknown == 255] = 0
    image_copy = image.copy()
    cv2.watershed(image_copy, markers)
    image_copy[markers == -1] = [255, 0, 0]
    return image_copy

def apply_kmeans(image, k, attempts):
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    return segmented_image.reshape(image.shape)

# --- STREAMLIT APP ---

st.title("Interactive Image Filtering and Segmentation")
st.write("Upload an image and apply different filters and segmentation algorithms. Adjust parameters interactively!")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    st.sidebar.title("Parameters")
    method = st.sidebar.selectbox("Choose a Method",
                                  ["Gaussian Blur", "Sobel Edge Detection", "Canny Edge Detection",
                                   "Otsu Thresholding", "Watershed Segmentation", "K-Means Clustering"])

    if method == "Gaussian Blur":
        kernel_size = st.sidebar.slider("Kernel Size", 3, 15, step=2, value=5)
        result = apply_gaussian_blur(image, kernel_size)
        st.image(result, channels="BGR", caption="Gaussian Blur Applied")

    elif method == "Sobel Edge Detection":
        ksize = st.sidebar.slider("Kernel Size", 3, 15, step=2, value=3)
        result = apply_sobel(image, ksize)
        st.image(result, caption="Sobel Edge Detection Applied", use_column_width=True, clamp=True)

    elif method == "Canny Edge Detection":
        threshold1 = st.sidebar.slider("Threshold1", 50, 200, value=100)
        threshold2 = st.sidebar.slider("Threshold2", 100, 300, value=200)
        result = apply_canny(image, threshold1, threshold2)
        st.image(result, caption="Canny Edge Detection Applied", use_column_width=True)

    elif method == "Otsu Thresholding":
        result = apply_otsu(image)
        st.image(result, caption="Otsu Thresholding Applied", use_column_width=True, clamp=True)

    elif method == "Watershed Segmentation":
        result = apply_watershed(image)
        st.image(result, channels="BGR", caption="Watershed Segmentation Applied")

    elif method == "K-Means Clustering":
        k = st.sidebar.slider("Number of Clusters (K)", 2, 10, value=3)
        attempts = st.sidebar.slider("K-Means Attempts", 1, 10, value=5)
        result = apply_kmeans(image, k, attempts)
        st.image(result, channels="BGR", caption="K-Means Clustering Applied")

else:
    st.info("Please upload an image to get started.")