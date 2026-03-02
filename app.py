import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# 1. Настройка страницы
st.set_page_config(page_title="SafeGuard ИИ", layout="centered")
st.title("🛡️ SafeGuard ИИ: Система мониторинга")

# 2. Загрузка модели
@st.cache_resource
def load_model():
    try:
        return YOLO('best.onnx', task='detect')
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None

model = load_model()

if model:
    st.sidebar.header("Настройки ИИ")
    conf_val = st.sidebar.slider("Чувствительность (Confidence)", 0.1, 1.0, 0.4, 
                                  help="Снизьте, если ИИ не видит каски. Поднимите, если много ложных срабатываний.")
    
    st.sidebar.write("---")
    st.sidebar.write("**Классы в вашей модели:**")
    st.sidebar.write(list(model.names.values()))

    # --- ГЛАВНАЯ ЛОГИКА АНАЛИЗА ---
    def process_frame_logic(img_cv, model, conf):
        # imgsz=320 для скорости, iou=0.5 для исключения наложений
        results = model.predict(img_cv, conf=conf, imgsz=320, iou=0.5, verbose=False)
        boxes = results[0].boxes
        
        h_img, w_img = img_cv.shape[:2]
        people = []
        protection_boxes = []

        if len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id].lower()
                coords = box.xyxy[0].tolist() # [x1, y1, x2, y2]
                
                if 'person' in label or 'human' in label:
                    people.append(coords)
                else:
                    protection_boxes.append({'label': label, 'coords': coords})
                    # Рисуем каски/жилеты тонкими линиями
                    cv2.rectangle(img_cv, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), (0, 255, 0), 2)

        # Считаем результаты
        safe_count = 0
        violation_count = 0

        for p in people:
            px1, py1, px2, py2 = p
            is_protected = False
            
            # Ищем, есть ли защита, которая пересекается с ВЕРХНЕЙ частью человека (головой)
            for prot in protection_boxes:
                rx1, ry1, rx2, ry2 = prot['coords']
                # Проверка: центр защиты должен быть внутри рамки человека
                mid_x = (rx1 + rx2) / 2
                mid_y = (ry1 + ry2) / 2
                if (px1 < mid_x < px2) and (py1 < mid_y < py2):
                    is_protected = True
                    break
            
            if is_protected:
                safe_count += 1
                cv2.rectangle(img_cv, (int(px1), int(py1)), (int(px2), int(py2)), (255, 255, 255), 1)
            else:
                violation_count += 1
                # Выделяем нарушителя КРАСНЫМ
                head_h = int((py2 - py1) * 0.25) # Зона головы
                cv2.rectangle(img_cv, (int(px1), int(py1)), (int(px2), int(py1 + head_h)), (0, 0, 255), 4)
                cv2.putText(img_cv, "NO PPE", (int(px1), int(py1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.rectangle(img_cv, (int(px1), int(py1)), (int(px2), int(py2)), (0, 0, 255), 1)

        # РИСУЕМ ТАБЛО СО СТАТИСТИКОЙ
        overlay = img_cv.copy()
        cv2.rectangle(overlay, (0, 0), (300, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, img_cv, 0.5, 0, img_cv)
        
        cv2.putText(img_cv, f"SAFE: {safe_count}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img_cv, f"VIOLATIONS: {violation_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return img_cv, safe_count, violation_count

    # --- КЛАСС ДЛЯ ВИДЕО ---
    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.conf = conf_val
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            processed, _, _ = process_frame_logic(img, model, self.conf)
            return av.VideoFrame.from_ndarray(processed, format="bgr24")

    # --- ИНТЕРФЕЙС ---
    tab1, tab2 = st.tabs(["🎥 Видео (LIVE)", "📁 Фото (Анализ)"])

    with tab1:
        webrtc_ctx = webrtc_streamer(
            key="ppe-monitoring",
            video_processor_factory=VideoProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": {"width": 640, "height": 480, "frameRate": 15}, "audio": False},
            async_processing=True,
        )
        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.conf = conf_val

    with tab2:
        up_img = st.file_uploader("Загрузите фото для проверки", type=['jpg', 'jpeg', 'png'])
        if up_img:
            img_in = Image.open(up_img)
            img_cv = cv2.cvtColor(np.array(img_in.convert("RGB")), cv2.COLOR_RGB2BGR)
            res_cv, safe, bad = process_frame_logic(img_cv, model, conf_val)
            
            # Вывод текстового вердикта под фото
            st.image(cv2.cvtColor(res_cv, cv2.COLOR_BGR2RGB), use_column_width=True)
            c1, c2 = st.columns(2)
            c1.metric("В защите", safe)
            c2.metric("Нарушители", bad, delta=-bad if bad > 0 else 0, delta_color="inverse")
            
            if bad > 0:
                st.error(f"Внимание! На объекте обнаружено {bad} чел. без СИЗ!")
            else:
                st.success("Все сотрудники в безопасности.")
else:
    st.error("Ошибка запуска нейросети.")
