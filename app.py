import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

# 1. Настройка страницы
st.set_page_config(page_title="SafeGuard PRO", layout="centered")
st.title("🛡️ SafeGuard ИИ v5.0")

# 2. Загрузка модели ONNX
@st.cache_resource
def load_session():
    # Использование CPUExecutionProvider для стабильности в облаке
    return ort.InferenceSession("best.onnx", providers=['CPUExecutionProvider'])

session = load_session()

# !!! ВНИМАНИЕ: Проверь порядок классов в своем metadata.json !!!
# Если порядок другой - просто переставь их в этом списке
classes = ['Helmet', 'No-Helmet', 'No-Vest', 'Person', 'Vest']

# --- ПРОФЕССИОНАЛЬНАЯ ОБРАБОТКА (NMS) ---
def detect_and_draw(img_bgr, conf_threshold):
    h_orig, w_orig = img_bgr.shape[:2]
    
    # Подготовка (YOLOv8 требует 640x640 и Float32)
    blob = cv2.resize(img_bgr, (640, 640))
    blob = blob.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1) # HWC -> CHW
    blob = np.expand_dims(blob, 0)

    # Запуск модели
    outputs = session.run(None, {session.get_inputs()[0].name: blob})
    output = outputs[0][0] # Матрица [классы+4, 8400]

    # Списки для фильтрации NMS
    boxes = []
    scores = []
    class_ids = []

    # Разбор вывода YOLOv8
    # YOLOv8 выводит: [x, y, w, h, class0, class1, ...]
    for i in range(output.shape[1]):
        classes_scores = output[4:, i]
        class_id = np.argmax(classes_scores)
        score = classes_scores[class_id]

        if score > conf_threshold:
            cx, cy, w, h = output[:4, i]
            
            # Масштабируем координаты к оригиналу
            x1 = int((cx - w/2) * (w_orig / 640))
            y1 = int((cy - h/2) * (h_orig / 640))
            width = int(w * (w_orig / 640))
            height = int(h * (h_orig / 640))

            boxes.append([x1, y1, width, height])
            scores.append(float(score))
            class_ids.append(class_id)

    # Фильтрация NMS (чтобы не было 100 рамок на одном человеке)
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, 0.45)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]] if class_ids[i] < len(classes) else "Unknown"
            conf = scores[i]

            # Настройка цвета
            if "no" in label.lower() or label.lower() == "person":
                color = (0, 0, 255) # Красный для нарушений и людей без пометок
            else:
                color = (0, 255, 0) # Зеленый для касок и жилетов

            # Рисуем рамку
            cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color, 3)
            
            # Текст подписи
            text = f"{label.upper()} {int(conf*100)}%"
            cv2.putText(img_bgr, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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
st.sidebar.write("---")
st.sidebar.write("Классы модели:", classes)

tab1, tab2 = st.tabs(["🎥 Живое видео", "📷 Анализ фото"])

with tab1:
    RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    webrtc_ctx = webrtc_streamer(
        key="ppe-vision",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": {"width": 640, "height": 480, "frameRate": 20}, "audio": False},
        async_processing=True,
    )
    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.conf_threshold = conf_val

with tab2:
    img_file = st.file_uploader("Загрузите изображение", type=["jpg", "png", "jpeg"])
    cam_file = st.camera_input("Или сделайте мгновенное фото")
    
    active_file = img_file if img_file else cam_file

    if active_file:
        input_image = Image.open(active_file)
        img_bgr = cv2.cvtColor(np.array(input_image.convert("RGB")), cv2.COLOR_RGB2BGR)
        
        with st.spinner('Анализирую...'):
            result_bgr = detect_and_draw(img_bgr, conf_val)
            st.image(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)
