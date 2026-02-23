import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# 1. Настройка страницы
st.set_page_config(page_title="SafeGuard PRO", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #0f172a; color: white; }
    .stMarkdown { text-align: center; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ SafeGuard ИИ")
st.write("Система контроля безопасности v5.0")

# 2. Загрузка модели
@st.cache_resource
def load_model():
    # Загружаем модель один раз и кешируем
    model = YOLO('best.onnx', task='detect')
    return model

model = load_model()

# 3. Боковая панель
st.sidebar.header("Параметры")
conf_val = st.sidebar.slider("Чувствительность", 0.1, 1.0, 0.5)
st.sidebar.write("---")
st.sidebar.info("В режиме 'Камера' снимок анализируется мгновенно. Это самый надежный способ для мобильных устройств.")

# --- ФУНКЦИЯ ОБРАБОТКИ ---
def process_result(img, model, conf):
    # Превращаем фото в массив
    img_rgb = np.array(img.convert("RGB"))
    img_cv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    # Запуск ИИ (imgsz=320 для скорости)
    results = model.predict(img, conf=conf, imgsz=320, verbose=False)
    boxes = results[0].boxes
    
    if len(boxes) == 0:
        return img_rgb, 0

    people = []
    protection_boxes = []

    # Разбираем объекты
    for box in boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id].lower()
        coords = box.xyxy[0].tolist()
        
        if 'person' in label or 'human' in label:
            people.append(coords)
        else:
            protection_boxes.append(coords)
            # Рисуем рамку самой защиты
            cv2.rectangle(img_cv, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), (0, 255, 0), 3)
            cv2.putText(img_cv, label.upper(), (int(coords[0]), int(coords[1]-10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Проверка каждого человека
    violations = 0
    for p in people:
        px1, py1, px2, py2 = p
        is_protected = False
        for prot in protection_boxes:
            rx1, ry1, rx2, ry2 = prot
            # Проверка пересечения
            if not (rx2 < px1 or rx1 > px2 or ry2 < py1 or ry1 > py2):
                is_protected = True
                break
        
        if not is_protected:
            violations += 1
            # Рисуем красное предупреждение
            cv2.putText(img_cv, "NO PROTECTION", (int(px1), int(py1-15)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.rectangle(img_cv, (int(px1), int(py1)), (int(px2), int(py2)), (0, 0, 255), 2)
        else:
            # Зеленая рамка вокруг защищенного человека
            cv2.rectangle(img_cv, (int(px1), int(py1)), (int(px2), int(py2)), (0, 255, 0), 1)

    return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), violations

# 4. ИНТЕРФЕЙС ВКЛАДОК
tab1, tab2 = st.tabs(["📷 Анализ через камеру", "📁 Загрузить файл"])

with tab1:
    # Самый стабильный способ работы с камерой в Streamlit
    cam_img = st.camera_input("Наведите камеру и сделайте фото")
    if cam_img:
        with st.spinner('Нейросеть анализирует...'):
            img_pil = Image.open(cam_img)
            res_img, count = process_result(img_pil, model, conf_val)
            st.image(res_img, use_column_width=True)
            if count > 0:
                st.error(f"⚠️ ОБНАРУЖЕНО НАРУШЕНИЙ: {count}")
            else:
                st.success("✅ Все сотрудники в средствах защиты")

with tab2:
    up_img = st.file_uploader("Выберите фото из галереи", type=['jpg', 'png', 'jpeg'])
    if up_img:
        img_pil = Image.open(up_img)
        res_img, count = process_result(img_pil, model, conf_val)
        st.image(res_img, use_column_width=True)
        if count > 0:
            st.warning(f"Найдено нарушений: {count}")
