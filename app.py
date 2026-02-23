import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Настройка интерфейса
st.set_page_config(page_title="SafeGuard PRO", layout="centered") # layout="centered" сужает страницу
st.title("🛡️ SafeGuard ИИ")
st.write("Система мониторинга СИЗ")

# Загрузка модели
@st.cache_resource
def load_model():
    return YOLO('best.onnx', task='detect')

model = load_model()

# Настройки в боковой панели
conf_val = st.sidebar.slider("Чувствительность ИИ", 0.1, 1.0, 0.5)

# Вкладки для режимов
tab1, tab2 = st.tabs(["📷 Сделать фото", "📁 Загрузить файл"])

with tab1:
    img_file = st.camera_input("Наведите камеру")
    if img_file is not None:
        img = Image.open(img_file)
        results = model.predict(img, conf=conf_val)
        res_plotted = results[0].plot()
        
        # Уменьшаем фото: создаем колонки (1/4 - 2/4 - 1/4)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(res_plotted, caption='Результат анализа', width=400) # Здесь можно менять ширину (width)

with tab2:
    uploaded_file = st.file_uploader("Выберите изображение...", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        results = model.predict(img, conf=conf_val)
        res_plotted = results[0].plot()
        
        # Уменьшаем фото: выводим в центральной колонке
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(res_plotted, caption='Результат анализа фото', width=400) # width=400 — это размер в пикселях
