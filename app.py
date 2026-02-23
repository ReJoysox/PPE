import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# Настройка интерфейса
st.set_page_config(page_title="SafeGuard PRO", layout="centered")
st.title("🛡️ SafeGuard ИИ")
st.write("Система контроля промышленной безопасности")

# Загрузка модели
@st.cache_resource
def load_model():
    return YOLO('best.onnx', task='detect')

model = load_model()

# Настройки
conf_val = st.sidebar.slider("Чувствительность ИИ", 0.1, 1.0, 0.5)

def process_and_draw(img, model, conf):
    # Превращаем фото в формат для OpenCV
    img_array = np.array(img)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    h_img, w_img, _ = img_cv.shape

    # Запускаем нейросеть
    results = model.predict(img, conf=conf)
    boxes = results[0].boxes

    # Словари для хранения найденных объектов
    people = []
    protection = []

    # Разбираем объекты по категориям
    for box in boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id].lower()
        coords = box.xyxy[0].tolist() # [x1, y1, x2, y2]
        
        if label == 'person':
            people.append(coords)
        else:
            protection.append({'label': label, 'coords': coords})

    # Логика: проверяем каждого человека
    for p in people:
        px1, py1, px2, py2 = p
        has_protection = False
        
        # Проверяем, есть ли защита внутри или рядом с рамкой человека
        for prot in protection:
            rx1, ry1, rx2, ry2 = prot['coords']
            # Если рамка защиты пересекается с рамкой человека
            if not (rx2 < px1 or rx1 > px2 or ry2 < py1 or ry1 > py2):
                has_protection = True
                # Рисуем рамку защиты (зеленая)
                cv2.rectangle(img_cv, (int(rx1), int(ry1)), (int(rx2), int(ry2)), (0, 255, 0), 3)
                cv2.putText(img_cv, prot['label'].upper(), (int(rx1), int(ry1)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Если защиты нет — рисуем предупреждение над головой
        if not has_protection:
            # Вычисляем зону головы (верхняя часть рамки человека)
            head_y = int(py1)
            cv2.rectangle(img_cv, (int(px1), head_y), (int(px2), int(py1 + (py2-py1)*0.2)), (0, 0, 255), 2)
            cv2.putText(img_cv, "!!! NO PROTECTION !!!", (int(px1), head_y - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            # Рисуем красную рамку вокруг человека, чтобы выделить нарушителя
            cv2.rectangle(img_cv, (int(px1), int(py1)), (int(px2), int(py2)), (0, 0, 255), 1)

    return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

# Вкладки
tab1, tab2 = st.tabs(["📷 Сделать фото", "📁 Загрузить файл"])

with tab1:
    img_file = st.camera_input("Наведите камеру")
    if img_file is not None:
        img = Image.open(img_file)
        processed_img = process_and_draw(img, model, conf_val)
        st.image(processed_img, width=500)

with tab2:
    uploaded_file = st.file_uploader("Выберите изображение...", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        processed_img = process_and_draw(img, model, conf_val)
        st.image(processed_img, width=500)
