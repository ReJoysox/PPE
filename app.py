import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="SafeGuard PRO | ONNX Web", layout="wide")

st.title("🛡️ SafeGuard ИИ: Промышленный мониторинг")
st.write("Запуск модели **YOLOv8 ONNX** в облаке")

@st.cache_resource
def load_model():
    # Загружаем твой файл model.onnx
    return YOLO('model.onnx', task='detect')

model = load_model()

conf_val = st.sidebar.slider("Уверенность ИИ", 0.1, 1.0, 0.5)

tab1, tab2 = st.tabs(["📷 Камера", "🖼️ Загрузка фото"])

with tab1:
    img_file = st.camera_input("Сделайте фото для анализа")
    if img_file is not None:
        img = Image.open(img_file)
        results = model.predict(img, conf=conf_val)
        res_plotted = results[0].plot()
        st.image(res_plotted, caption='Результат анализа')

with tab2:
    uploaded_file = st.file_uploader("Выберите файл...", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        results = model.predict(img, conf=conf_val)
        res_plotted = results[0].plot()
        st.image(res_plotted, use_column_width=True)
