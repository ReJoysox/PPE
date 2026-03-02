import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

# Настройка страницы
st.set_page_config(page_title="SafeGuard PRO", layout="centered")
st.title("🛡️ SafeGuard ИИ: Видео и Фото")

# 1. Загрузка модели ONNX
@st.cache_resource
def load_session():
    return ort.InferenceSession("best.onnx", providers=['CPUExecutionProvider'])

session = load_session()
# Список классов (проверь свои в metadata.json)
classes = ['Helmet', 'No-Helmet', 'No-Vest', 'Person', 'Vest']

# --- ОБЩАЯ ФУНКЦИЯ ОБРАБОТКИ (ДЛЯ ВИДЕО И ФОТО) ---
def detect_and_draw(img_bgr, conf_threshold):
    h_orig, w_orig = img_bgr.shape[:2]
    
    # Подготовка для YOLO (640x640)
    blob = cv2.resize(img_bgr, (640, 640))
    blob = blob.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)
    blob = np.expand_dims(blob, 0)

    # Инференс
    outputs = session.run(None, {session.get_inputs()[0].name: blob})
    output = outputs[0][0]

    # Отрисовка
    for i in range(output.shape[1]):
        scores = output[4:, i]
        class_id = np.argmax(scores)
        score = scores[class_id]

        if score > conf_threshold:
            cx, cy, w, h = output[:4, i]
            x1 = int((cx - w/2) * (w_orig / 640))
            y1 = int((cy - h/2) * (h_orig / 640))
            x2 = int((cx + w/2) * (w_orig / 640))
            y2 = int((cy + h/2) * (h_orig / 640))

            label = classes[class_id] if class_id < len(classes) else "Object"
            
            if label.lower() == 'person':
                # Тонкая рамка для человека
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 255, 255), 1)
            else:
                color = (0, 255, 0) if "no" not in label.lower() else (0, 0, 255)
                # Рисуем рамку СИЗ
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 3)
                # Пишем название
                cv2.putText(img_bgr, f"{label.upper()} {int(score*100)}%", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                # Если нарушение - добавляем плашку
                if "no" in label.lower():
                    cv2.putText(img_bgr, "!!! DANGER !!!", (x1, y1-35), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    return img_bgr

# --- КЛАСС ДЛЯ ВИДЕОПОТОКА ---
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.conf_threshold = 0.5

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        processed_img = detect_and_draw(img, self.conf_threshold)
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

# --- ИНТЕРФЕЙС ---
st.sidebar.header("Настройки ИИ")
conf_val = st.sidebar.slider("Чувствительность", 0.1, 1.0, 0.5)

tab1, tab2 = st.tabs(["🎥 Живое видео (LIVE)", "📷 Анализ фото"])

with tab1:
    st.write("Нажмите START для запуска мониторинга")
    RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    
    webrtc_ctx = webrtc_streamer(
        key="ppe-live",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": {"width": 640, "height": 480, "frameRate": 15}, "audio": False},
        async_processing=True,
    )
    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.conf_threshold = conf_val

with tab2:
    mode = st.radio("Источник фото:", ["Загрузить файл", "Сделать снимок сейчас"])
    
    img_file = None
    if mode == "Загрузить файл":
        img_file = st.file_uploader("Выберите изображение...", type=["jpg", "png", "jpeg"])
    else:
        img_file = st.camera_input("Сделайте фото для анализа")

    if img_file:
        # Обработка фото
        input_image = Image.open(img_file)
        img_bgr = cv2.cvtColor(np.array(input_image.convert("RGB")), cv2.COLOR_RGB2BGR)
        
        with st.spinner('Нейросеть анализирует снимок...'):
            result_bgr = detect_and_draw(img_bgr, conf_val)
            result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            st.image(result_rgb, caption="Результат анализа", use_column_width=True)
