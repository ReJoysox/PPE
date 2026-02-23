import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# Настройка страницы
st.set_page_config(page_title="SafeGuard FAST", layout="centered")
st.title("🛡️ SafeGuard ИИ: Оптимизированный LIVE")

# Загрузка модели
@st.cache_resource
def load_model():
    return YOLO('best.onnx', task='detect')

model = load_model()

# Снижаем порог уверенности чуть-чуть для скорости
conf_val = 0.5

# --- ОПТИМИЗИРОВАННАЯ ЛОГИКА ---
def fast_logic(img, model):
    # imgsz=320 делает работу нейросети в 4 раза быстрее, чем стандартные 640
    results = model.predict(img, conf=conf_val, imgsz=320, verbose=False)
    boxes = results[0].boxes
    
    if len(boxes) == 0:
        return img

    people = []
    protection = []

    for box in boxes:
        c = box.xyxy[0].tolist()
        label = model.names[int(box.cls[0])].lower()
        if 'person' in label or 'human' in label:
            people.append(c)
        else:
            protection.append(c)
            # Рисуем тонкие рамки (толстые линии тормозят отрисовку)
            cv2.rectangle(img, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (0, 255, 0), 2)

    for p in people:
        px1, py1, px2, py2 = p
        is_safe = any(not (r[2] < px1 or r[0] > px2 or r[3] < py1 or r[1] > py2) for r in protection)
        
        if is_safe:
            cv2.rectangle(img, (int(px1), int(py1)), (int(px2), int(py2)), (255, 255, 255), 1)
        else:
            cv2.putText(img, "ALERT", (int(px1), int(py1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.rectangle(img, (int(px1), int(py1)), (int(px2), int(py2)), (0, 0, 255), 1)
    
    return img

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # Обработка
        processed = fast_logic(img, model)
        return av.VideoFrame.from_ndarray(processed, format="bgr24")

# Конфигурация WebRTC (используем Google сервера для стабильности)
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

webrtc_streamer(
    key="fast-ppe",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIG,
    # ХАК №1: Принудительно снижаем разрешение видео с камеры до 320p
    media_stream_constraints={
        "video": {
            "width": {"ideal": 480},
            "height": {"ideal": 320},
            "frameRate": {"ideal": 15}
        },
        "audio": False,
    },
    async_processing=True, # ХАК №2: Не блокируем поток при обработке
)

st.info("Для максимальной скорости используйте хорошее освещение и формат ONNX.")
