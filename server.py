import io
import os
from typing import List, Tuple, Dict

import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from ultralytics import YOLO

MODEL_PATH = "best.onnx"

# Что считаем "защитой"
REQUIRE_HELMET = True
REQUIRE_VEST = False  # если хочешь требовать и жилет, поставь True

CONF = 0.25           # базовый порог детекции (можно поднимать)
IOU = 0.5             # NMS внутри YOLO
IMGSZ = 640           # можно 640, можно 512/320 быстрее

app = FastAPI(title="SafeGuard PPE API")

# Разрешаем запросы со страницы GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # можно сузить до https://<user>.github.io
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Не найден {MODEL_PATH}. Положи best.onnx рядом с server.py")

model = YOLO(MODEL_PATH, task="detect")

def _find_class_ids(names: Dict[int, str]):
    """
    Пытаемся найти нужные классы по названию.
    Если в твоей модели названия другие, всё равно часто ловится по подстрокам.
    """
    person_ids, helmet_ids, vest_ids, nohelmet_ids, novest_ids = [], [], [], [], []
    for cid, name in names.items():
        n = name.lower()
        if "person" in n or "human" in n:
            person_ids.append(cid)
        if "helmet" in n or "hardhat" in n:
            if "no" in n:
                nohelmet_ids.append(cid)
            else:
                helmet_ids.append(cid)
        if "vest" in n:
            if "no" in n:
                novest_ids.append(cid)
            else:
                vest_ids.append(cid)
    return person_ids, helmet_ids, vest_ids, nohelmet_ids, novest_ids

PERSON_IDS, HELMET_IDS, VEST_IDS, NOHELMET_IDS, NOVEST_IDS = _find_class_ids(model.names)

def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
    area_b = max(0, bx2-bx1) * max(0, by2-by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union

def center_in(person_xyxy, obj_xyxy, zone="full") -> bool:
    px1, py1, px2, py2 = person_xyxy
    ox1, oy1, ox2, oy2 = obj_xyxy
    cx = (ox1 + ox2) / 2
    cy = (oy1 + oy2) / 2

    if zone == "head":
        # верхняя часть тела: 0..35%
        head_y2 = py1 + 0.35*(py2-py1)
        return (px1 <= cx <= px2) and (py1 <= cy <= head_y2)

    return (px1 <= cx <= px2) and (py1 <= cy <= py2)

def draw_label(img, x, y, text, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    y0 = max(0, y - th - 10)
    cv2.rectangle(img, (x, y0), (x + tw + 10, y0 + th + 10), color, -1)
    cv2.putText(img, text, (x + 5, y0 + th + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

def analyze(image_bgr: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    Возвращает: (картинка_с_рамками, safe_count, bad_count)
    Рисуем ОДНУ рамку на человека.
    """
    res = model.predict(image_bgr, conf=CONF, iou=IOU, imgsz=IMGSZ, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return image_bgr, 0, 0

    boxes = res.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)

    persons = []
    helmets = []
    vests = []
    nohelmets = []
    novests = []

    for b, c in zip(xyxy, cls):
        b = b.tolist()
        if c in PERSON_IDS:
            persons.append(b)
        elif c in HELMET_IDS:
            helmets.append(b)
        elif c in VEST_IDS:
            vests.append(b)
        elif c in NOHELMET_IDS:
            nohelmets.append(b)
        elif c in NOVEST_IDS:
            novests.append(b)

    # Доп. NMS только по persons (убрать "5 людей на одном")
    persons_sorted = sorted(persons, key=lambda bb: (bb[2]-bb[0])*(bb[3]-bb[1]), reverse=True)
    filtered_persons = []
    for p in persons_sorted:
        if all(iou_xyxy(p, fp) < 0.5 for fp in filtered_persons):
            filtered_persons.append(p)

    safe = 0
    bad = 0

    for p in filtered_persons:
        has_helmet = any(center_in(p, h, zone="head") for h in helmets)
        has_vest = any(center_in(p, v, zone="full") for v in vests)
        has_nohelmet = any(center_in(p, nh, zone="head") for nh in nohelmets)
        has_novest = any(center_in(p, nv, zone="full") for nv in novests)

        # Правило статуса
        ok = True
        if REQUIRE_HELMET:
            ok = ok and has_helmet and not has_nohelmet
        if REQUIRE_VEST:
            ok = ok and has_vest and not has_novest

        # Если в модели нет no- классов, тогда отсутствие helmet/vest = нарушение
        if REQUIRE_HELMET and (len(NOHELMET_IDS) == 0) and (not has_helmet):
            ok = False
        if REQUIRE_VEST and (len(NOVEST_IDS) == 0) and (not has_vest):
            ok = False

        x1,y1,x2,y2 = map(int, p)
        if ok:
            safe += 1
            color = (34,197,94)   # green
            cv2.rectangle(image_bgr, (x1,y1), (x2,y2), color, 3)
            draw_label(image_bgr, x1, y1, "В ЗАЩИТЕ", color)
        else:
            bad += 1
            color = (68,68,239)   # red-ish BGR? -> actually (B,G,R)
            color = (0,0,255)
            cv2.rectangle(image_bgr, (x1,y1), (x2,y2), color, 3)
            draw_label(image_bgr, x1, y1, "БЕЗ СИЗ", color)
            # голова красным
            head_y2 = int(y1 + 0.25*(y2-y1))
            cv2.rectangle(image_bgr, (x1,y1), (x2,head_y2), color, 3)

    # табло
    cv2.rectangle(image_bgr, (0,0), (260,70), (0,0,0), -1)
    cv2.putText(image_bgr, f"SAFE: {safe}", (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(image_bgr, f"DANGER: {bad}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    return image_bgr, safe, bad

@app.get("/health")
def health():
    return {
        "ok": True,
        "model": MODEL_PATH,
        "names": model.names,
        "person_ids": PERSON_IDS,
        "helmet_ids": HELMET_IDS,
        "vest_ids": VEST_IDS,
        "nohelmet_ids": NOHELMET_IDS,
        "novest_ids": NOVEST_IDS
    }

@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    content = await file.read()
    img_arr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "cannot decode image"}, status_code=400)

    out_img, safe, bad = analyze(img)
    ok, png = cv2.imencode(".png", out_img)
    if not ok:
        return JSONResponse({"error": "cannot encode output"}, status_code=500)

    headers = {
        "X-SAFE": str(safe),
        "X-BAD": str(bad),
        "Access-Control-Expose-Headers": "X-SAFE, X-BAD"
    }
    return Response(content=png.tobytes(), media_type="image/png", headers=headers)
