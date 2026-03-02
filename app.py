import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

st.set_page_config(page_title="SafeGuard PRO", layout="centered")
st.title("🛡️ Инспектор СИЗ(система индивидуальной безопасности)")

@st.cache_resource
def load_model():
    try:
        return YOLO('best.onnx', task='detect')
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None

model = load_model()

if model:
    st.sidebar.header("Настройки")
    # Уменьшаем порог по умолчанию, чтобы ИИ видел человека, даже если он прикрыт
    conf_val = st.sidebar.slider("Чувствительность", 0.1, 1.0, 0.3)
    
    st.sidebar.write("---")
    st.sidebar.write("Классы модели:", list(model.names.values()))

    def process_frame_logic(img_cv, model, conf):
        # iou=0.3 помогает лучше разделять близко стоящих людей
        results = model.predict(img_cv, conf=conf, imgsz=320, iou=0.3, verbose=False)
        boxes = results[0].boxes
        
        people = []
        protection = []
        direct_violations = []

        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id].lower()
            coords = box.xyxy[0].tolist()
            conf_score = float(box.conf[0])
            
            # Если модель напрямую видит класс "БЕЗ КАСКИ" (обычно в PPE датасетах это есть)
            if 'no-' in label:
                direct_violations.append({'label': label, 'coords': coords})
            elif 'person' in label or 'human' in label:
                people.append(coords)
            else:
                protection.append(coords)
                # Рисуем найденную защиту (Зеленым)
                cv2.rectangle(img_cv, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), (0, 255, 0), 2)

        safe_count = 0
        violation_count = 0
        reported_people = [] # Чтобы не считать одного и того же человека дважды

        # 1. Сначала обрабатываем прямые нарушения (no-helmet, no-vest)
        for violation in direct_violations:
            violation_count += 1
            v_coords = violation['coords']
            cv2.rectangle(img_cv, (int(v_coords[0]), int(v_coords[1])), (int(v_coords[2]), int(v_coords[3])), (0, 0, 255), 3)
            cv2.putText(img_cv, f"ALARM: {violation['label'].upper()}", (int(v_coords[0]), int(v_coords[1]-10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 2. Проверяем людей методом пересечения (если прямых нарушений не нашли)
        for p in people:
            px1, py1, px2, py2 = p
            is_protected = False
            
            # Проверяем, нет ли уже в этой зоне прямого нарушения
            already_violated = any(not (v['coords'][2] < px1 or v['coords'][0] > px2 or v['coords'][3] < py1 or v['coords'][1] > py2) for v in direct_violations)
            if already_violated:
                continue

            for prot_coords in protection:
                rx1, ry1, rx2, ry2 = prot_coords
                # Центр каски должен быть в рамке человека
                if (px1 < (rx1+rx2)/2 < px2) and (py1 < (ry1+ry2)/2 < py2):
                    is_protected = True
                    break
            
            if is_protected:
                safe_count += 1
                cv2.rectangle(img_cv, (int(px1), int(py1)), (int(px2), int(py2)), (255, 255, 255), 1)
            else:
                violation_count += 1
                cv2.putText(img_cv, "NO PPE DETECTED", (int(px1), int(py1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.rectangle(img_cv, (int(px1), int(py1)), (int(px2), int(py2)), (0, 0, 255), 2)

        # РИСУЕМ ТАБЛО
        cv2.rectangle(img_cv, (0, 0), (220, 70), (0, 0, 0), -1)
        cv2.putText(img_cv, f"SAFE: {safe_count}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img_cv, f"DANGER: {violation_count}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return img_cv, safe_count, violation_count

    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.conf = conf_val
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            processed, _, _ = process_frame_logic(img, model, self.conf)
            return av.VideoFrame.from_ndarray(processed, format="bgr24")

    tab1, tab2 = st.tabs(["🎥 LIVE ВИДЕО", "📁 АНАЛИЗ ФОТО"])

    with tab1:
        webrtc_streamer(
            key="ppe-safe",
            video_processor_factory=VideoProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": {"width": 640, "height": 480, "frameRate": 15}, "audio": False},
            async_processing=True,
        )

    with tab2:
        up_img = st.file_uploader("Загрузите фото")
        if up_img:
            img_cv = cv2.cvtColor(np.array(Image.open(up_img).convert("RGB")), cv2.COLOR_RGB2BGR)
            res_cv, safe, bad = process_frame_logic(img_cv, model, conf_val)
            st.image(cv2.cvtColor(res_cv, cv2.COLOR_BGR2RGB), use_column_width=True)
            st.write(f"### Результат: {safe} в защите, {bad} нарушителей.")
