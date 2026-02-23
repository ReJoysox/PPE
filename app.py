import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Настройка интерфейса
st.set_page_config(page_title="SafeGuard PRO", layout="centered")
st.title("🛡️ SafeGuard ИИ")
st.write("Система мониторинга СИЗ")

# Загрузка модели
@st.cache_resource
def load_model():
    return YOLO('best.onnx', task='detect')

model = load_model()

# Настройки в боковой панели
conf_val = st.sidebar.slider("Чувствительность ИИ", 0.1, 1.0, 0.5)

# --- ЛОГИКА ФИЛЬТРАЦИИ ---
# Получаем все ID классов, кроме тех, что называются 'person'
# Это гарантирует, что мы не будем рисовать рамку вокруг человека
target_classes = [id for id, name in model.names.items() if name.lower() != 'person']

# Вкладки для режимов
tab1, tab2 = st.tabs(["📷 Сделать фото", "📁 Загрузить файл"])

with tab1:
    img_file = st.camera_input("Наведите камеру")
    if img_file is not None:
        img = Image.open(img_file)
        
        # Предсказываем только выбранные классы СИЗ (без person)
        results = model.predict(img, conf=conf_val, classes=target_classes)
        res_plotted = results[0].plot()
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(res_plotted, caption='Результат анализа', width=400)

with tab2:
    uploaded_file = st.file_uploader("Выберите изображение...", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        
        # Предсказываем только выбранные классы СИЗ (без person)
        results = model.predict(img, conf=conf_val, classes=target_classes)
        res_plotted = results[0].plot()
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(res_plotted, caption='Результат анализа фото', width=400)
