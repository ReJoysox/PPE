import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, Vid<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SafeGuard ИИ | Контроль СИЗ</title>
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
    <style>
        :root { --p: #3b82f6; --bg: #0f172a; --card: #1e293b; --s: #22c55e; --d: #ef4444; }
        body { font-family: 'Segoe UI', sans-serif; background: var(--bg); color: white; margin: 0; padding: 20px; display: flex; flex-direction: column; align-items: center; }
        .container { width: 100%; max-width: 900px; background: var(--card); padding: 25px; border-radius: 20px; box-shadow: 0 15px 40px rgba(0,0,0,0.5); }
        h1 { color: var(--p); margin: 0 0 10px 0; text-align: center; }
        
        #monitor { position: relative; width: 100%; background: #000; border-radius: 15px; overflow: hidden; border: 3px solid #334155; display: flex; justify-content: center; align-items: center; min-height: 480px; }
        canvas { max-width: 100%; max-height: 600px; display: block; }
        video { display: none; }

        .controls { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 20px 0; }
        .btn { padding: 15px; border: none; border-radius: 10px; font-weight: bold; color: white; cursor: pointer; transition: 0.3s; }
        .btn-live { background: var(--p); }
        .btn-photo { background: #8b5cf6; text-align: center; }
        .btn:hover { opacity: 0.8; transform: translateY(-2px); }

        .settings { background: #0f172a; padding: 15px; border-radius: 10px; margin-bottom: 20px; }
        .settings label { display: flex; justify-content: space-between; font-size: 0.9rem; margin-bottom: 10px; }
        input[type=range] { width: 100%; cursor: pointer; }

        .stats { display: flex; justify-content: space-around; background: #0f172a; padding: 15px; border-radius: 12px; margin-top: 10px; }
        .stat-item { text-align: center; flex: 1; }
        .stat-val { display: block; font-size: 2.2rem; font-weight: bold; }
        
        #status { text-align: center; color: #38bdf8; font-weight: bold; margin-bottom: 10px; }
        input[type="file"] { display: none; }
    </style>
</head>
<body>

<div class="container">
    <h1>🛡️ SafeGuard ИИ v8.0</h1>
    <div id="status">Загрузка модели best.onnx...</div>

    <div class="settings">
        <label>
            <span>Чувствительность ИИ:</span>
            <span id="conf-val" style="color:var(--p); font-weight:bold;">30%</span>
        </label>
        <input type="range" id="conf-range" min="10" max="90" step="5" value="30">
    </div>

    <div id="monitor">
        <video id="video" autoplay muted playsinline></video>
        <canvas id="canvas"></canvas>
    </div>

    <div class="controls">
        <button class="btn btn-live" onclick="startCamera()">🎥 ЖИВОЙ ЭФИР</button>
        <label class="btn btn-photo" style="display:flex; align-items:center; justify-content:center;">
            📁 АНАЛИЗ ФОТО
            <input type="file" id="file-input" accept="image/*">
        </label>
    </div>

    <div class="stats">
        <div class="stat-item" style="border-right: 2px solid #1e293b;">
            <span style="color:#94a3b8; font-size: 0.8rem;">С СИЗ</span>
            <span id="count-safe" class="stat-val" style="color:var(--s)">0</span>
        </div>
        <div class="stat-item">
            <span style="color:#94a3b8; font-size: 0.8rem;">БЕЗ СИЗ</span>
            <span id="count-bad" class="stat-val" style="color:var(--d)">0</span>
        </div>
    </div>
</div>

<script>
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const status = document.getElementById('status');
    const confRange = document.getElementById('conf-range');
    const confValLabel = document.getElementById('conf-val');

    let session;
    let isLive = false;
    let labels = ["Helmet", "No-Helmet", "No-Vest", "Person", "Vest"];

    // 1. ЗАГРУЗКА МОДЕЛИ
    async function init() {
        try {
            session = await ort.InferenceSession.create('./best.onnx', { executionProviders: ['wasm'] });
            status.innerText = "✅ МОДЕЛЬ ЗАГРУЖЕНА. ГОТОВ К РАБОТЕ.";
            status.style.color = "#22c55e";
        } catch (e) {
            status.innerText = "❌ ОШИБКА: Файл best.onnx не найден в папке!";
            status.style.color = "#ef4444";
        }
    }

    // 2. КАМЕРА
    async function startCamera() {
        if (!session) return;
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
        video.srcObject = stream;
        isLive = true;
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            detectLoop();
        };
    }

    async function detectLoop() {
        if (!isLive) return;
        await runInference(video);
        requestAnimationFrame(detectLoop);
    }

    // 3. ФОТО
    document.getElementById('file-input').onchange = (e) => {
        const file = e.target.files[0];
        if (!file) return;
        isLive = false;
        if (video.srcObject) video.srcObject.getTracks().forEach(t => t.stop());
        
        const img = new Image();
        img.src = URL.createObjectURL(file);
        img.onload = async () => {
            canvas.width = img.width;
            canvas.height = img.height;
            await runInference(img);
        };
    };

    function sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    // 4. ГЛАВНЫЙ ДВИЖОК
    async function runInference(source) {
        if (!session) return;

        const [w, h] = [640, 640];
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = w; tempCanvas.height = h;
        const tCtx = tempCanvas.getContext('2d');
        tCtx.drawImage(source, 0, 0, w, h);
        const imgData = tCtx.getImageData(0, 0, w, h).data;

        const input = new Float32Array(3 * w * h);
        for (let i = 0; i < w * h; i++) {
            input[i] = imgData[i * 4] / 255.0; 
            input[i + w*h] = imgData[i * 4 + 1] / 255.0; 
            input[i + 2*w*h] = imgData[i * 4 + 2] / 255.0; 
        }

        const tensor = new ort.Tensor('float32', input, [1, 3, w, h]);
        const outputs = await session.run({ [session.inputNames[0]]: tensor });
        const output = outputs[session.outputNames[0]];

        parseAndDraw(output.data, output.dims, source);
    }

    function parseAndDraw(data, dims, source) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(source, 0, 0, canvas.width, canvas.height);

        const threshold = parseInt(confRange.value) / 100;
        let detections = [];

        let numAnchors = dims[2];
        let numClasses = dims[1] - 4;

        for (let i = 0; i < numAnchors; i++) {
            let maxProb = 0;
            let classId = -1;

            for (let c = 0; c < numClasses; c++) {
                let raw_val = data[(4 + c) * numAnchors + i];
                let prob = sigmoid(raw_val);
                if (prob > maxProb) { maxProb = prob; classId = c; }
            }

            if (maxProb > threshold) {
                let cx = data[0 * numAnchors + i];
                let cy = data[1 * numAnchors + i];
                let bw = data[2 * numAnchors + i];
                let bh = data[3 * numAnchors + i];

                let x = (cx - bw / 2) * (canvas.width / 640);
                let y = (cy - bh / 2) * (canvas.height / 640);
                let w = bw * (canvas.width / 640);
                let h = bh * (canvas.height / 640);

                detections.push({ x, y, w, h, prob: maxProb, classId });
            }
        }

        // NMS - УБИРАЕМ ДУБЛИКАТЫ РАМОК
        detections.sort((a, b) => b.prob - a.prob);
        const finalBoxes = [];
        while (detections.length > 0) {
            const best = detections.shift();
            finalBoxes.push(best);
            detections = detections.filter(box => iou(best, box) < 0.45);
        }

        let safeCount = 0;
        let badCount = 0;

        finalBoxes.forEach(box => {
            const labelName = labels[box.classId] ? labels[box.classId] : "Object";
            const labelLower = labelName.toLowerCase();
            
            if (labelLower === "person" || labelLower === "human") return;

            const isBad = labelLower.includes("no");
            const color = isBad ? "#ef4444" : "#22c55e";
            const text = isBad ? "НАРУШИТЕЛЬ" : "В ЗАЩИТЕ";
            
            if (isBad) badCount++; else safeCount++;

            ctx.strokeStyle = color;
            ctx.lineWidth = 4;
            ctx.strokeRect(box.x, box.y, box.w, box.h);

            ctx.fillStyle = color;
            ctx.fillRect(box.x, box.y > 25 ? box.y - 25 : 0, box.w, 25);
            
            ctx.fillStyle = "white";
            ctx.font = "bold 14px Arial";
            ctx.fillText(`${text} ${Math.round(box.prob * 100)}%`, box.x + 5, box.y > 25 ? box.y - 7 : 18);
        });

        document.getElementById('count-safe').innerText = safeCount;
        document.getElementById('count-bad').innerText = badCount;
    }

    function iou(b1, b2) {
        const x1 = Math.max(b1.x, b2.x), y1 = Math.max(b1.y, b2.y);
        const x2 = Math.min(b1.x + b1.w, b2.x + b2.w), y2 = Math.min(b1.y + b1.h, b2.y + b2.h);
        const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        const union = b1.w * b1.h + b2.w * b2.h - inter;
        return inter / union;
    }

    confRange.addEventListener('input', (e) => {
        confValLabel.innerText = e.target.value + '%';
    });

    init();
</script>
</body>
</html>eoProcessorBase, RTCConfiguration
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
