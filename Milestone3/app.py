# app.py  — MILESTONE 3 (FRONTEND ONLY)

import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
from inference import process_and_predict

st.set_page_config(page_title="PCB Defect Detection", layout="wide")
st.title("🛠️ PCB Defect Detection & Classification System (Milestone 3)")

# --------- UPLOAD UI (Module 5) ---------
st.subheader("Step 1: Upload Images")

col1, col2 = st.columns(2)

with col1:
    template_file = st.file_uploader("Upload Template PCB", type=["jpg","png"])
with col2:
    test_file = st.file_uploader("Upload Test PCB", type=["jpg","png"])

# --------- LIVE INFERENCE (Module 6 call) ---------
if template_file and test_file:

    t_bytes = np.asarray(bytearray(template_file.read()), dtype=np.uint8)
    test_bytes = np.asarray(bytearray(test_file.read()), dtype=np.uint8)

    template_img = cv2.imdecode(t_bytes, cv2.IMREAD_COLOR)
    test_img = cv2.imdecode(test_bytes, cv2.IMREAD_COLOR)

    st.subheader("Processing... ⏳")
    start = time.time()

    mask, annotated, preds, logs = process_and_predict(template_img, test_img)

    elapsed = round(time.time() - start, 2)
    st.success(f"Processing Time: {elapsed} seconds")

    # --------- DISPLAY RESULTS (Module 5) ---------
    st.markdown("---")
    st.subheader("Input Images")

    c1, c2 = st.columns(2)
    with c1:
        st.image(template_img, caption="Template PCB", channels="BGR")
    with c2:
        st.image(test_img, caption="Test PCB", channels="BGR")

    st.markdown("---")
    st.subheader("Generated Defect Mask")
    st.image(mask, caption="Defect Mask", clamp=True)

    st.markdown("---")
    st.subheader("Annotated PCB with Bounding Boxes & Labels")
    st.image(annotated, channels="BGR")

    # --------- LIVE PREDICTION TABLE ---------
    st.subheader("Live Predictions (This Image Only)")

    live_df = pd.DataFrame({
        "Defect_Index": list(range(len(preds))),
        "Predicted_Defect": preds
    })

    st.dataframe(live_df)

    # --------- LIGHT LOGS (Module 6 deliverable) ---------
    st.subheader("Backend Logs (for this run)")
    st.json(logs)

else:
    st.info("Upload both template and test images to start prediction.")
