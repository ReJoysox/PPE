import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import numpy as np
import onnxruntime as ort

# Настройка страницы
st.set_page_config(page_title="SafeGuard LIVE", layout="centered")
st.title("🛡️ SafeGuard ИИ: Real-Time")

# 1. Загрузка модели ONNX
@st.cache_resource
def load_session():
    # Используем CPU-движок для стабильности на сервере
    return ort.InferenceSession("best.onnx", providers=['CPUExecutionProvider'])

session = load_session()
# Классы (замени на свои, если они другие)
classes = ['Helmet', 'No-Helmet', 'No-Vest', 'Person', 'Vest']

# 2. Логика обработки кадра
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.conf_threshold = 0.5

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h_orig, w_orig = img.shape[:2]

        # Подготовка кадра (640x640 для YOLO)
        input_img = cv2.resize(img, (640, 640))
        input_img = input_img.astype(np.float32) / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_img = np.expand_dims(input_img, 0)

        # Инференс
        outputs = session.run(None, {session.get_inputs()[0].name: input_img})
        output = outputs[0][0] # [84, 8400]

        # Постобработка (отрисовка)
        for i in range(output.shape[1]):
            scores = output[4:, i]
            class_id = np.argmax(scores)
            score = scores[class_id]

            if score > self.conf_threshold:
                cx, cy, w, h = output[:4, i]
                
                # Масштабирование координат
                x1 = int((cx - w/2) * (w_orig / 640))
                y1 = int((cy - h/2) * (h_orig / 640))
                x2 = int((cx + w/2) * (w_orig / 640))
                y2 = int((cy + h/2) * (h_orig / 640))

                label = classes[class_id] if class_id < len(classes) else "Object"
                
                # Логика: если 'person' - рисуем тонкую рамку, если СИЗ - жирную
                if label.lower() == 'person':
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
                else:
                    color = (0, 255, 0) if "no" not in label.lower() else (0, 0, 255)
                    # Если нарушения (no-helmet), пишем крупно
                    if "no" in label.lower():
                        cv2.putText(img, "WARNING: NO PPE", (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 3)
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
                    cv2.putText(img, f"{label} {int(score*100)}%", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 3. Настройка WebRTC (используем публичные STUN-серверы Google)
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="safe-guard-live",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640},
            "height": {"ideal": 480},
            "frameRate": {"ideal": 15} # Ограничиваем FPS для стабильности сервера
        },
        "audio": False,
    },
    async_processing=True,
)

st.sidebar.info("Для переключения камер на телефоне используйте настройки браузера или разверните видео.")
