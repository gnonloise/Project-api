
import cv2
import streamlit as st
from ultralytics import YOLO

# Définition du chemin vers le model
model_path = "weights/yolov8n.pt"

st.set_page_config(
   page_title= " Object detection using YOLOV8"
,  page_icon= " 🤖 " 
,  layout= "wide"
,  initial_sidebar_state= "expanded"

)

with st.sidebar :
    st.header("Vidéo/Webcam Config")
    uploaded_file = st.sidebar.file_uploader("Choose the video file" , type= ["mp4"])
    source_vid = st.sidebar._selectbox(
        "Or select webcam" , ["Webcam"]
    )

model = YOLO(model_path)

if uploaded_file is not None :
    source_vid = 0

elif source_vid == "Webcam":
    source_vid = 0

if source_vid is not None : 
    vid_cap = cv2.VideoCapture (str(source_vid))
    if st.sidebar.button('Detect objects'):
        st_frame = st.empty()
        while vid_cap.isOpened():
            success, image = cv2.resize(image,(720, 405))
            res = model.predict(image)
            result_image = res[0].render()
            st_frame.image(result_image, caption= "Detection video" , channels= "BGR")
    else:
        vid_cap.release()
        break
