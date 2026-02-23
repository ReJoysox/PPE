import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# Настройка страницы
st.set_page_config(page_title="SafeGuard PRO", layout="centered")
st.title("🛡️ SafeGuard ИИ")

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
    # Показываем классы модели в боковой панели для отладки
    st.sidebar.write("### Классы в вашей модели:")
    st.sidebar.write(list(model.names.values()))
    
    conf_val = st.sidebar.slider("Чувствительность", 0.1, 1.0, 0.4)

    def process_frame(img):
        # Конвертация PIL -> OpenCV (BGR)
        img_cv = cv2.cvtColor(np.array(img), list(cv2.COLOR_RGB2BGR if len(np.array(img).shape)==3 else cv2.COLOR_GRAY2BGR))
        
        # Запуск ИИ
        results = model.predict(img, conf=conf_val)
        boxes = results[0].boxes
        
        found_person = False
        found_protection = False

        if len(boxes) == 0:
            return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

        # 1. Сначала ищем всю защиту и рисуем её
        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id].lower()
            xyxy = box.xyxy[0].tolist()
            
            # Если это НЕ человек, рисуем зеленую рамку защиты
            if 'person' not in label and 'human' not in label:
                found_protection = True
                cv2.rectangle(img_cv, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 3)
                cv2.putText(img_cv, label.upper(), (int(xyxy[0]), int(xyxy[1]-10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                found_person = True
                # Рисуем тонкую рамку вокруг человека
                cv2.rectangle(img_cv, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 255, 255), 1)

        # 2. Логика предупреждения
        # Если нашли человека, но не нашли защиту в кадре
        if found_person and not found_protection:
            for box in boxes:
                label = model.names[int(box.cls[0])].lower()
                if 'person' in label or 'human' in label:
                    xyxy = box.xyxy[0].tolist()
                    # Пишем КРАСНЫМ над головой
                    cv2.putText(img_cv, "!!! NO PROTECTION !!!", (int(xyxy[0]), int(xyxy[1]-15)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                    # Выделяем голову красным прямоугольником
                    head_h = int((xyxy[3] - xyxy[1]) * 0.25)
                    cv2.rectangle(img_cv, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[1] + head_h)), (0, 0, 255), 2)

        return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    # Интерфейс вкладок
    t1, t2 = st.tabs(["🎥 Камера", "📁 Загрузка"])

    with t1:
        cam_img = st.camera_input("Сделайте снимок")
        if cam_img:
            res = process_frame(Image.open(cam_img))
            st.image(res, width=500)

    with t2:
        up_img = st.file_uploader("Загрузите фото", type=['jpg', 'png', 'jpeg'])
        if up_img:
            res = process_frame(Image.open(up_img))
            st.image(res, width=500)
else:
    st.error("Файл best.onnx не найден в репозитории!")
