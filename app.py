import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# Настройка страницы
st.set_page_config(page_title="SafeGuard PRO", layout="centered")
st.markdown("<style>.stApp {background-color: #0f172a; color: white;}</style>", unsafe_allow_html=True)

st.title("🛡️ SafeGuard ИИ")
st.write("Система контроля СИЗ (v5.0 Stable)")

# Загрузка модели
@st.cache_resource
def load_model():
    # Загружаем твой файл best.onnx
    return YOLO('best.onnx', task='detect')

model = load_model()

# Настройки в сайдбаре
conf_val = st.sidebar.slider("Чувствительность ИИ", 0.1, 1.0, 0.5)

def process_image(img, model, conf):
    # Превращаем PIL Image в BGR массив для OpenCV
    img_rgb = np.array(img.convert("RGB"))
    img_cv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    # Запуск ИИ
    results = model.predict(img, conf=conf, imgsz=320, verbose=False)
    boxes = results[0].boxes
    
    if len(boxes) == 0:
        return img_rgb, 0

    people = []
    protection = []

    # 1. Сортируем объекты
    for box in boxes:
        c = box.xyxy[0].tolist()
        label = model.names[int(box.cls[0])].lower()
        if 'person' in label or 'human' in label:
            people.append(c)
        else:
            protection.append({'label': label, 'coords': c})
            # Рисуем саму защиту (зеленым)
            cv2.rectangle(img_cv, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (0, 255, 0), 3)
            cv2.putText(img_cv, label.upper(), (int(c[0]), int(c[1]-7)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 2. Проверяем каждого человека
    violations = 0
    for p in people:
        px1, py1, px2, py2 = p
        is_safe = False
        for prot in protection:
            rx1, ry1, rx2, ry2 = prot['coords']
            # Если рамка защиты пересекается с рамкой человека
            if not (rx2 < px1 or rx1 > px2 or ry2 < py1 or ry1 > py2):
                is_safe = True
                break
        
        if not is_safe:
            violations += 1
            # Красная зона головы
            head_h = int((py2 - py1) * 0.25)
            cv2.rectangle(img_cv, (int(px1), int(py1)), (int(px2), int(py1 + head_h)), (0, 0, 255), 3)
            cv2.putText(img_cv, "NO PROTECTION", (int(px1), int(py1-15)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.rectangle(img_cv, (int(px1), int(py1)), (int(px2), int(py2)), (0, 0, 255), 1)
        else:
            # Если всё ок - белая рамка
            cv2.rectangle(img_cv, (int(px1), int(py1)), (int(px2), int(py2)), (255, 255, 255), 1)

    return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), violations

# Интерфейс
t1, t2 = st.tabs(["📷 Сделать фото", "📁 Загрузить файл"])

with t1:
    cam_img = st.camera_input("Наведите камеру")
    if cam_img:
        res, count = process_image(Image.open(cam_img), model, conf_val)
        st.image(res, width=500)
        if count > 0: st.error(f"Нарушений: {count}")
        else: st.success("Безопасно")

with t2:
    file_img = st.file_uploader("Выберите фото", type=['jpg','jpeg','png'])
    if file_img:
        res, count = process_image(Image.open(file_img), model, conf_val)
        st.image(res, width=500)
