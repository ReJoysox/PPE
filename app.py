import threading
from pathlib import Path

import av
import cv2
import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer


# -------------------- Настройки/пути --------------------
ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "best.onnx"
CLASSES_PATH = ROOT / "classes.txt"
IMGSZ = 320
IOU_THRES = 0.3

st.set_page_config(page_title="SafeGuard PRO", layout="centered")
st.title("SafeGuard ИИ: Система мониторинга")


# -------------------- Глобальная конфигурация (для webrtc потока) --------------------
_conf_lock = threading.Lock()
CURRENT_CONF = 0.3


def set_conf(v: float):
    global CURRENT_CONF
    with _conf_lock:
        CURRENT_CONF = float(v)


def get_conf() -> float:
    with _conf_lock:
        return float(CURRENT_CONF)


# -------------------- Утилиты --------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def letterbox(img_bgr, new_shape=320, color=(114, 114, 114)):
    h, w = img_bgr.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_w = new_shape - nw
    pad_h = new_shape - nh
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return out, r, (left, top)


def iou_xyxy(a, b):
    # a: (4,), b: (N,4)
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)
    inter = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = np.maximum(0, bx2 - bx1) * np.maximum(0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union


def nms(boxes, scores, iou_thres=0.3):
    if len(boxes) == 0:
        return []

    boxes = boxes.astype(np.float32)
    scores = scores.astype(np.float32)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = iou_xyxy(boxes[i], boxes[rest])
        order = rest[ious < iou_thres]

    return keep


@st.cache_resource
def load_class_names():
    if not CLASSES_PATH.exists():
        return None
    lines = CLASSES_PATH.read_text(encoding="utf-8").splitlines()
    names = [l.strip() for l in lines if l.strip()]
    return names if names else None


@st.cache_resource
def load_onnx_session():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Не найден файл модели: {MODEL_PATH}")
    sess = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    return sess, input_name


def yolo_onnx_predict(img_bgr, sess, input_name, class_names, conf_thres=0.3, imgsz=320, iou_thres=0.3):
    h0, w0 = img_bgr.shape[:2]
    img_lb, r, (padx, pady) = letterbox(img_bgr, new_shape=imgsz)

    img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
    x = img_rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None, ...]  # 1x3xHxW

    outputs = sess.run(None, {input_name: x})
    pred = outputs[0]

    # Приводим к виду (num_preds, n)
    if pred.ndim == 3:
        pred = pred[0]
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T

    n = pred.shape[1]
    if class_names is None:
        # грубая оценка: считаем, что obj нет (YOLOv8 style)
        nc = max(1, n - 4)
        class_names = [f"class{i}" for i in range(nc)]
    else:
        nc = len(class_names)

    # Пытаемся понять есть ли objectness
    has_obj = (n == nc + 5)

    # Если похоже на logits — применим sigmoid к "хвосту"
    tail = pred[:, 4:] if not has_obj else pred[:, 4:]
    if np.nanmax(tail) > 1.0:
        pred[:, 4:] = sigmoid(pred[:, 4:])

    boxes_xywh = pred[:, :4].copy()

    # Если координаты похожи на нормализованные (0..1) — домножим на imgsz
    if np.nanmax(boxes_xywh) <= 1.5:
        boxes_xywh[:, [0, 2]] *= imgsz
        boxes_xywh[:, [1, 3]] *= imgsz

    # xywh -> xyxy (в letterbox координатах)
    x_c, y_c, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    x1 = x_c - w / 2
    y1 = y_c - h / 2
    x2 = x_c + w / 2
    y2 = y_c + h / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    if has_obj:
        obj = pred[:, 4]
        cls_probs = pred[:, 5:]
        cls_id = np.argmax(cls_probs, axis=1)
        cls_score = cls_probs[np.arange(cls_probs.shape[0]), cls_id]
        scores = obj * cls_score
    else:
        cls_probs = pred[:, 4:]
        cls_id = np.argmax(cls_probs, axis=1)
        scores = cls_probs[np.arange(cls_probs.shape[0]), cls_id]

    mask = scores >= conf_thres
    boxes = boxes[mask]
    scores = scores[mask]
    cls_id = cls_id[mask]

    if boxes.shape[0] == 0:
        return []

    keep = nms(boxes, scores, iou_thres=iou_thres)
    boxes = boxes[keep]
    scores = scores[keep]
    cls_id = cls_id[keep]

    # Возвращаем координаты в оригинальное изображение
    dets = []
    for b, sc, ci in zip(boxes, scores, cls_id):
        x1, y1, x2, y2 = b
        x1 = (x1 - padx) / r
        y1 = (y1 - pady) / r
        x2 = (x2 - padx) / r
        y2 = (y2 - pady) / r

        x1 = float(np.clip(x1, 0, w0 - 1))
        y1 = float(np.clip(y1, 0, h0 - 1))
        x2 = float(np.clip(x2, 0, w0 - 1))
        y2 = float(np.clip(y2, 0, h0 - 1))

        label = class_names[int(ci)].lower() if int(ci) < len(class_names) else f"class{int(ci)}"
        dets.append({"label": label, "coords": [x1, y1, x2, y2], "score": float(sc)})

    return dets


def process_frame_logic(img_cv, sess, input_name, class_names, conf):
    dets = yolo_onnx_predict(
        img_cv, sess, input_name, class_names,
        conf_thres=conf, imgsz=IMGSZ, iou_thres=IOU_THRES
    )

    people = []
    protection = []
    direct_violations = []

    for d in dets:
        label = d["label"]
        coords = d["coords"]

        if "no-" in label or "no_" in label or "without" in label:
            direct_violations.append({"label": label, "coords": coords})
        elif "person" in label or "human" in label:
            people.append(coords)
        else:
            protection.append(coords)
            cv2.rectangle(
                img_cv,
                (int(coords[0]), int(coords[1])),
                (int(coords[2]), int(coords[3])),
                (0, 255, 0),
                2,
            )

    safe_count = 0
    violation_count = 0

    # 1) Прямые нарушения
    for violation in direct_violations:
        violation_count += 1
        v = violation["coords"]
        cv2.rectangle(img_cv, (int(v[0]), int(v[1])), (int(v[2]), int(v[3])), (0, 0, 255), 3)
        cv2.putText(
            img_cv,
            f"ALARM: {violation['label'].upper()}",
            (int(v[0]), int(v[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    # 2) Проверка людей по пересечению с защитой
    for p in people:
        px1, py1, px2, py2 = p

        # Если уже есть прямое нарушение в зоне человека — не считаем второй раз
        already_violated = any(
            not (v["coords"][2] < px1 or v["coords"][0] > px2 or v["coords"][3] < py1 or v["coords"][1] > py2)
            for v in direct_violations
        )
        if already_violated:
            continue

        is_protected = False
        for prot in protection:
            rx1, ry1, rx2, ry2 = prot
            cx, cy = (rx1 + rx2) / 2, (ry1 + ry2) / 2
            if (px1 < cx < px2) and (py1 < cy < py2):
                is_protected = True
                break

        if is_protected:
            safe_count += 1
            cv2.rectangle(img_cv, (int(px1), int(py1)), (int(px2), int(py2)), (255, 255, 255), 1)
        else:
            violation_count += 1
            cv2.putText(
                img_cv,
                "NO PPE DETECTED",
                (int(px1), int(py1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            cv2.rectangle(img_cv, (int(px1), int(py1)), (int(px2), int(py2)), (0, 0, 255), 2)

    # Табло
    cv2.rectangle(img_cv, (0, 0), (220, 70), (0, 0, 0), -1)
    cv2.putText(img_cv, f"SAFE: {safe_count}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img_cv, f"DANGER: {violation_count}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return img_cv, safe_count, violation_count


# -------------------- UI --------------------
try:
    class_names = load_class_names()
    sess, input_name = load_onnx_session()
except Exception as e:
    st.error(f"Ошибка инициализации: {e}")
    st.stop()

st.sidebar.header("Настройки")
conf_val = st.sidebar.slider("Чувствительность", 0.05, 1.0, 0.3, 0.05)
set_conf(conf_val)

st.sidebar.write("---")
st.sidebar.write("Классы:", class_names if class_names else "classes.txt не найден (используются class0..)")


class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        conf = get_conf()
        processed, _, _ = process_frame_logic(img, sess, input_name, class_names, conf)
        return av.VideoFrame.from_ndarray(processed, format="bgr24")


tab1, tab2 = st.tabs(["LIVE ВИДЕО", "АНАЛИЗ ФОТО"])

with tab1:
    webrtc_streamer(
        key="ppe-safe",
        video_processor_factory=VideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": {"width": 640, "height": 480, "frameRate": 15}, "audio": False},
        async_processing=True,
    )

with tab2:
    up_img = st.file_uploader("Загрузите фото", type=["jpg", "jpeg", "png"])
    if up_img:
        img_cv = cv2.cvtColor(np.array(Image.open(up_img).convert("RGB")), cv2.COLOR_RGB2BGR)
        res_cv, safe, bad = process_frame_logic(img_cv, sess, input_name, class_names, conf_val)
        st.image(cv2.cvtColor(res_cv, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.write(f"### Результат: {safe} в защите, {bad} нарушителей.")
