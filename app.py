import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# Настройка страницы
st.set_page_config(page_title="SafeGuard PRO", layout="centered")
st.title("🛡️ SafeGuard ИИ")
st.write("Система контроля средств индивидуальной защиты")

# Загрузка модели
@st.cache_resource
def load_model():
    try:
        model = YOLO('best.onnx', task='detect')
        return model
    except Exception as e:
        st.error(f"Ошибка загрузки best.onnx: {e}")
        return None

model = load_model()

if model:
    # Сайдбар с информацией
    st.sidebar.write("### Обнаружение классов:")
    st.sidebar.write(list(model.names.values()))
    conf_val = st.sidebar.slider("Чувствительность", 0.1, 1.0, 0.4)

    def process_frame(img):
        # 1. Подготовка изображения
        # Конвертируем в RGB, затем в массив numpy, затем в BGR для OpenCV
        img_rgb = np.array(img.convert("RGB"))
        img_cv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # 2. Запуск нейросети
        results = model.predict(img, conf=conf_val)
        boxes = results[0].boxes
        
        if len(boxes) == 0:
            return img_rgb

        # Списки для объектов
        people = []
        protection_boxes = []

        # 3. Сортируем найденные объекты
        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id].lower()
            coords = box.xyxy[0].tolist() # [x1, y1, x2, y2]
            
            if 'person' in label or 'human' in label:
                people.append(coords)
            else:
                protection_boxes.append(coords)
                # Рисуем зеленую рамку для самой защиты (каска/жилет)
                cv2.rectangle(img_cv, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), (0, 255, 0), 3)
                cv2.putText(img_cv, label.upper(), (int(coords[0]), int(coords[1]-10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 4. Проверяем каждого человека на наличие защиты
        for p in people:
            px1, py1, px2, py2 = p
            is_protected = False
            
            # Проверяем, пересекается ли какая-либо защита с рамкой этого человека
            for prot in protection_boxes:
                rx1, ry1, rx2, ry2 = prot
                # Простая проверка пересечения прямоугольников
                if not (rx2 < px1 or rx1 > px2 or ry2 < py1 or ry1 > py2):
                    is_protected = True
                    break
            
            if is_protected:
                # Человек в защите — рисуем тонкую белую рамку
                cv2.rectangle(img_cv, (int(px1), int(py1)), (int(px2), int(py2)), (255, 255, 255), 1)
            else:
                # Человека БЕЗ защиты — выделяем КРАСНЫМ
                # 1. Надпись над головой
                cv2.putText(img_cv, "NO PROTECTION", (int(px1), int(py1-15)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # 2. Рамка вокруг головы
                head_h = int((py2 - py1) * 0.25)
                cv2.rectangle(img_cv, (int(px1), int(py1)), (int(px2), int(py1 + head_h)), (0, 0, 255), 3)
                # 3. Рамка вокруг всего человека
                cv2.rectangle(img_cv, (int(px1), int(py1)), (int(px2), int(py2)), (0, 0, 255), 1)

        return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    # Интерфейс вкладок
    t1, t2 = st.tabs(["🎥 Сделать фото", "📁 Загрузить файл"])

    with t1:
        cam_img = st.camera_input("Наведите камеру")
        if cam_img:
            res = process_frame(Image.open(cam_img))
            st.image(res, width=500)

    with t2:
        up_img = st.file_uploader("Выберите фото", type=['jpg', 'png', 'jpeg'])
        if up_img:
            res = process_frame(Image.open(up_img))
            st.image(res, width=500)
else:
    st.error("Модель не загружена. Проверьте наличие файла best.onnx")
