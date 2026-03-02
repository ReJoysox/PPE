import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# 1. Настройка страницы
st.set_page_config(page_title="SafeGuard PRO", layout="centered")
st.title("🛡️ SafeGuard ИИ: Live + Photo")

# 2. Загрузка модели (кешируем, чтобы не грузить каждый раз)
@st.cache_resource
def load_model():
    try:
        # Загружаем вашу модель best.onnx
        return YOLO('best.onnx', task='detect')
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None

model = load_model()

if model:
    st.sidebar.write("### Настройки обнаружения:")
    conf_val = st.sidebar.slider("Чувствительность", 0.1, 1.0, 0.5)
    st.sidebar.write("Классы модели:", list(model.names.values()))

    # --- ЕДИНАЯ ЛОГИКА ОБРАБОТКИ ---
    def process_frame_logic(img_cv, model, conf):
        # imgsz=320 ускоряет работу в 3 раза для Live-видео
        results = model.predict(img_cv, conf=conf, imgsz=320, verbose=False)
        boxes = results[0].boxes
        
        if len(boxes) == 0:
            return img_cv

        people = []
        protection_boxes = []

        # 1. Разделяем людей и защиту
        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id].lower()
            coords = box.xyxy[0].tolist() # [x1, y1, x2, y2]
            
            if 'person' in label or 'human' in label:
                people.append(coords)
            else:
                protection_boxes.append(coords)
                # Рисуем саму защиту (каску/жилет) зеленым
                cv2.rectangle(img_cv, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), (0, 255, 0), 2)
                cv2.putText(img_cv, label.upper(), (int(coords[0]), int(coords[1]-5)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 2. Проверяем каждого человека
        for p in people:
            px1, py1, px2, py2 = p
            is_protected = False
            
            # Проверяем, пересекается ли любая защита с рамкой этого человека
            for prot in protection_boxes:
                rx1, ry1, rx2, ry2 = prot
                # Если рамка защиты находится внутри или сильно пересекается с рамкой человека
                if not (rx2 < px1 or rx1 > px2 or ry2 < py1 or ry1 > py2):
                    is_protected = True
                    break
            
            if is_protected:
                # Если всё ок - рисуем тонкую белую рамку (чтобы не мешать)
                cv2.rectangle(img_cv, (int(px1), int(py1)), (int(px2), int(py2)), (255, 255, 255), 1)
            else:
                # НАРУШИТЕЛЬ: рисуем КРАСНУЮ зону головы и надпись
                # Вычисляем зону головы (верхние 25% рамки человека)
                head_height = int((py2 - py1) * 0.25)
                # Надпись
                cv2.putText(img_cv, "NO PROTECTION", (int(px1), int(py1-10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Красная рамка головы
                cv2.rectangle(img_cv, (int(px1), int(py1)), (int(px2), int(py1 + head_height)), (0, 0, 255), 3)
                # Красная рамка всего тела (тонкая)
                cv2.rectangle(img_cv, (int(px1), int(py1)), (int(px2), int(py2)), (0, 0, 255), 1)
        
        return img_cv

    # --- КЛАСС ДЛЯ ОБРАБОТКИ ВИДЕО WebRTC ---
    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.conf = conf_val

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            # Применяем логику
            processed = process_frame_logic(img, model, self.conf)
            return av.VideoFrame.from_ndarray(processed, format="bgr24")

    # --- ИНТЕРФЕЙС (ВКЛАДКИ) ---
    tab1, tab2 = st.tabs(["🎥 Живое видео (Live)", "📁 Анализ фото"])

    with tab1:
        st.info("Нажмите START для включения видео-мониторинга")
        webrtc_ctx = webrtc_streamer(
            key="ppe-live",
            video_processor_factory=VideoProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": {"width": 640, "height": 480, "frameRate": 15}, "audio": False},
            async_processing=True,
        )
        # Передаем значение слайдера в видео-процессор
        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.conf = conf_val

    with tab2:
        up_img = st.file_uploader("Загрузите фото для анализа", type=['jpg', 'jpeg', 'png'])
        if up_img:
            img_input = Image.open(up_img)
            # Конвертируем PIL -> BGR
            img_cv = cv2.cvtColor(np.array(img_input.convert("RGB")), cv2.COLOR_RGB2BGR)
            # Обрабатываем
            res_cv = process_frame_logic(img_cv, model, conf_val)
            # Выводим (BGR -> RGB)
            st.image(cv2.cvtColor(res_cv, cv2.COLOR_BGR2RGB), use_column_width=True)

else:
    st.error("Не удалось инициализировать систему ИИ.")
