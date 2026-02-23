import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# Настройка страницы
st.set_page_config(page_title="SafeGuard Mobile", layout="centered")
st.title("🛡️ SafeGuard ИИ: Mobile LIVE")

# Загрузка модели
@st.cache_resource
def load_model():
    return YOLO('best.onnx', task='detect')

model = load_model()

# --- БОКОВАЯ ПАНЕЛЬ ---
st.sidebar.header("Настройки")
# Выбор камеры для телефона
camera_option = st.sidebar.radio(
    "Выберите камеру:",
    ("Фронтальная (Selfie)", "Основная (Rear)"),
    index=0
)

# Переводим выбор в понятный для браузера формат
facing_mode = "user" if camera_option == "Фронтальная (Selfie)" else "environment"

conf_val = st.sidebar.slider("Чувствительность", 0.1, 1.0, 0.5)

# --- ЛОГИКА ОБРАБОТКИ ---
def process_logic(img, model):
    # Оптимизация imgsz=320 для скорости на мобильных устройствах
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
            cv2.rectangle(img, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (0, 255, 0), 2)

    for p in people:
        px1, py1, px2, py2 = p
        # Проверка пересечения
        is_safe = any(not (r[2] < px1 or r[0] > px2 or r[3] < py1 or r[1] > py2) for r in protection)
        
        color = (0, 255, 0) if is_safe else (0, 0, 255)
        text = "SAFE" if is_safe else "NO PPE"
        
        cv2.rectangle(img, (int(px1), int(py1)), (int(px2), int(py2)), color, 1)
        cv2.putText(img, text, (int(px1), int(py1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return img

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        processed = process_logic(img, model)
        return av.VideoFrame.from_ndarray(processed, format="bgr24")

# Конфигурация WebRTC
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Запуск стримера
ctx = webrtc_streamer(
    key="mobile-ppe",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIG,
    # ПЕРЕДАЕМ ПАРАМЕТРЫ КАМЕРЫ
    media_stream_constraints={
        "video": {
            "facingMode": facing_mode, # Выбор камеры здесь
            "width": {"ideal": 480},
            "height": {"ideal": 320},
            "frameRate": {"ideal": 15}
        },
        "audio": False,
    },
    async_processing=True,
)

if ctx.state.playing:
    st.success(f"Трансляция запущена: {camera_option}")
else:
    st.info("Выберите камеру в боковом меню и нажмите START")
