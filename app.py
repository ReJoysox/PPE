import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# Настройка страницы
st.set_page_config(page_title="SafeGuard LITE", layout="centered")
st.title("🛡️ SafeGuard ИИ: Оптимизированный LIVE")

# Загрузка модели
@st.cache_resource
def load_model():
    return YOLO('best.onnx', task='detect')

model = load_model()

# Настройки в боковой панели
st.sidebar.header("Настройки мобильной версии")
camera_option = st.sidebar.radio("Камера:", ("Селфи", "Основная"))
facing_mode = "user" if camera_option == "Селфи" else "environment"
conf_val = st.sidebar.slider("Чувствительность", 0.1, 1.0, 0.5)

# --- УЛЬТРА-ОПТИМИЗИРОВАННАЯ ЛОГИКА ---
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.last_results = None

    def recv(self, frame):
        self.frame_count += 1
        img = frame.to_ndarray(format="bgr24")

        # ХАК: Обрабатываем только каждый 5-й кадр, чтобы сервер не зависал
        if self.frame_count % 5 == 0:
            # imgsz=160 — экстремальное сжатие для ИИ (очень быстро)
            results = model.predict(img, conf=conf_val, imgsz=160, verbose=False)
            self.last_results = results[0].boxes
        
        # Если есть результаты с прошлого анализа — рисуем их
        if self.last_results is not None:
            people = []
            protection = []

            for box in self.last_results:
                c = box.xyxy[0].tolist()
                label = model.names[int(box.cls[0])].lower()
                if 'person' in label or 'human' in label:
                    people.append(c)
                else:
                    protection.append(c)
                    cv2.rectangle(img, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (0, 255, 0), 2)

            for p in people:
                px1, py1, px2, py2 = p
                is_safe = any(not (r[2] < px1 or r[0] > px2 or r[3] < py1 or r[1] > py2) for r in protection)
                color = (0, 255, 0) if is_safe else (0, 0, 255)
                cv2.rectangle(img, (int(px1), int(py1)), (int(px2), int(py2)), color, 1)
                if not is_safe:
                    cv2.putText(img, "NO PPE", (int(px1), int(py1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Стандартная конфигурация серверов Google
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

webrtc_streamer(
    key="mobile-fast",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={
        "video": {
            "facingMode": facing_mode,
            "width": {"max": 480}, # Ограничиваем размер кадра
            "frameRate": {"max": 20}
        },
        "audio": False,
    },
    async_processing=True, # Не ждать завершения ИИ для показа видео
)

st.warning("⚠️ Если видео зависло — нажмите STOP и снова START. Серверу нужно время 'прогреться'.")
