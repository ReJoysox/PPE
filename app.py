import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Настройка интерфейса
st.set_page_config(page_title="SafeGuard PRO", layout="wide")
st.title("🛡️ SafeGuard ИИ: Система мониторинга")
st.write("Модель: **YOLOv8 ONNX (best.onnx)**")

# Загрузка модели
@st.cache_resource
def load_model():
    # Используем твой файл best.onnx
    return YOLO('best.onnx', task='detect')

model = load_model()

# Настройки в боковой панели
conf_val = st.sidebar.slider("Чувствительность ИИ", 0.1, 1.0, 0.5)
st.sidebar.write("---")
st.sidebar.write("Проект подготовлен для конкурса «Взлет»")

# Вкладки для режимов
tab1, tab2 = st.tabs(["📷 Сделать фото", "📁 Загрузить файл"])

with tab1:
    img_file = st.camera_input("Наведите камеру на объект")
    if img_file is not None:
        img = Image.open(img_file)
        results = model.predict(img, conf=conf_val)
        res_plotted = results[0].plot()
        st.image(res_plotted, caption='Результат анализа', use_column_width=True)

with tab2:
    uploaded_file = st.file_uploader("Выберите изображение...", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        results = model.predict(img, conf=conf_val)
        res_plotted = results[0].plot()
        st.image(res_plotted, caption='Анализ загруженного фото', use_column_width=True)
