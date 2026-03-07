<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SafeGuard ИИ | Идеальная точность</title>
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
    <style>
        :root { --p: #3b82f6; --bg: #0f172a; --card: #1e293b; --s: #22c55e; --d: #ef4444; }
        body { font-family: 'Segoe UI', Tahoma, sans-serif; background: var(--bg); color: white; margin: 0; padding: 20px; display: flex; flex-direction: column; align-items: center; }
        .container { width: 100%; max-width: 900px; background: var(--card); padding: 25px; border-radius: 20px; box-shadow: 0 15px 40px rgba(0,0,0,0.5); }
        h1 { color: var(--p); margin: 0 0 15px; text-align: center; font-size: 1.8rem; }
        
        #monitor { position: relative; width: 100%; background: #000; border-radius: 12px; overflow: hidden; border: 2px solid #334155; display: flex; justify-content: center; align-items: center; min-height: 480px; }
        canvas { max-width: 100%; max-height: 600px; display: block; }
        video { display: none; }

        .controls { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 20px 0; }
        .btn { padding: 15px; border: none; border-radius: 10px; font-weight: bold; color: white; cursor: pointer; transition: 0.2s; font-size: 1rem; }
        .btn-live { background: var(--p); }
        .btn-photo { background: #8b5cf6; text-align: center; }
        .btn:hover { opacity: 0.8; transform: translateY(-2px); }

        .settings { background: #0f172a; padding: 15px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #334155; }
        .settings label { display: flex; justify-content: space-between; font-size: 1rem; margin-bottom: 10px; font-weight: bold;}
        input[type=range] { width: 100%; cursor: pointer; }

        .stats { display: flex; justify-content: space-around; background: #0f172a; padding: 15px; border-radius: 12px; margin-top: 10px; border: 1px solid #334155;}
        .stat-item { text-align: center; flex: 1; }
        .stat-val { display: block; font-size: 2.5rem; font-weight: bold; }
        
        #status { text-align: center; color: #38bdf8; font-weight: bold; margin-bottom: 10px; font-size: 0.9rem;}
        input[type="file"] { display: none; }
    </style>
</head>
<body>

<div class="container">
    <h1>🛡️ SafeGuard ИИ v10.0</h1>
    <div id="status">Загрузка файла best.onnx...</div>

    <div class="settings">
        <label>
            <span>Чувствительность ИИ:</span>
            <span id="conf-val" style="color:var(--p);">40%</span>
        </label>
        <input type="range" id="conf-range" min="5" max="95" step="5" value="40">
    </div>

    <div id="monitor">
        <video id="video" autoplay muted playsinline></video>
        <canvas id="canvas"></canvas>
    </div>

    <div class="controls">
        <button class="btn btn-live" onclick="startCamera()">🎥 ВКЛЮЧИТЬ КАМЕРУ</button>
        <label class="btn btn-photo" style="display:flex; align-items:center; justify-content:center;">
            📁 ЗАГРУЗИТЬ ФОТО
            <input type="file" id="file-input" accept="image/*">
        </label>
    </div>

    <div class="stats">
        <div class="stat-item" style="border-right: 2px solid #1e293b;">
            <span style="color:#94a3b8; font-size: 0.8rem;">ЛЮДЕЙ В ЗАЩИТЕ</span>
            <span id="count-safe" class="stat-val" style="color:var(--s)">0</span>
        </div>
        <div class="stat-item">
            <span style="color:#94a3b8; font-size: 0.8rem;">ЛЮДЕЙ БЕЗ СИЗ</span>
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
    
    // Стандартные названия классов. 
    const labels = ["Helmet", "No-Helmet", "No-Vest", "Person", "Vest", "Head"];

    // 1. ЗАГРУЗКА
    async function init() {
        try {
            session = await ort.InferenceSession.create('./best.onnx', { executionProviders: ['wasm'] });
            status.innerText = "✅ МОДЕЛЬ ЗАГРУЖЕНА. ГОТОВ К РАБОТЕ.";
            status.style.color = "#22c55e";
        } catch (e) {
            status.innerText = "❌ ОШИБКА: Файл best.onnx не найден!";
            status.style.color = "#ef4444";
        }
    }

    // 2. КАМЕРА
    async function startCamera() {
        if (!session) return alert("Подождите загрузки модели!");
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
            video.srcObject = stream;
            isLive = true;
            video.onloadedmetadata = () => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                detectLoop();
            };
        } catch (e) { alert("Ошибка доступа к камере."); }
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
        
        status.innerText = "Анализ фотографии...";
        status.style.color = "#38bdf8";

        const img = new Image();
        img.src = URL.createObjectURL(file);
        img.onload = async () => {
            canvas.width = img.width;
            canvas.height = img.height;
            await runInference(img);
            status.innerText = "✅ АНАЛИЗ ЗАВЕРШЕН";
            status.style.color = "#22c55e";
        };
    };

    // 4. ДВИЖОК
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
        
        parseAndDraw(outputs[session.outputNames[0]].data, outputs[session.outputNames[0]].dims, source);
    }

    // 5. УМНЫЙ ПАРСИНГ И ОТРИСОВКА (ИСПРАВЛЕНА ПРОБЛЕМА 100% и НАСЛОЕНИЙ)
    function parseAndDraw(data, dims, source) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(source, 0, 0, canvas.width, canvas.height);

        const threshold = parseInt(confRange.value) / 100;
        let rawBoxes = [];

        // Автоматически определяем структуру матрицы от YOLO
        const isTransposed = dims[1] === 8400;
        const numAnchors = isTransposed ? dims[1] : dims[2];
        const numElements = isTransposed ? dims[2] : dims[1];
        const numClasses = numElements - 4;

        // Достаем все рамки
        for (let i = 0; i < numAnchors; i++) {
            let maxProb = 0; let classId = -1;
            let cx, cy, bw, bh;

            if (isTransposed) {
                cx = data[i * numElements + 0]; cy = data[i * numElements + 1];
                bw = data[i * numElements + 2]; bh = data[i * numElements + 3];
                for (let c = 0; c < numClasses; c++) {
                    let prob = data[i * numElements + 4 + c];
                    if (prob > maxProb) { maxProb = prob; classId = c; }
                }
            } else {
                cx = data[0 * numAnchors + i]; cy = data[1 * numAnchors + i];
                bw = data[2 * numAnchors + i]; bh = data[3 * numAnchors + i];
                for (let c = 0; c < numClasses; c++) {
                    let prob = data[(4 + c) * numAnchors + i];
                    if (prob > maxProb) { maxProb = prob; classId = c; }
                }
            }

            // ФИЛЬТР ПОЛЗУНКА РАБОТАЕТ ТУТ
            if (maxProb > threshold) {
                let x = (cx - bw / 2) * (canvas.width / 640);
                let y = (cy - bh / 2) * (canvas.height / 640);
                let w = bw * (canvas.width / 640);
                let h = bh * (canvas.height / 640);
                rawBoxes.push({ x, y, w, h, prob: maxProb, classId });
            }
        }

        // ШАГ 1: NMS (Удаляем дубликаты одного и того же объекта)
        rawBoxes.sort((a, b) => b.prob - a.prob);
        let cleanBoxes = [];
        while (rawBoxes.length > 0) {
            let best = rawBoxes.shift();
            cleanBoxes.push(best);
            rawBoxes = rawBoxes.filter(box => iou(best, box) < 0.4);
        }

        // ШАГ 2: ГРУППИРОВКА РАМОК ПО ЛЮДЯМ (РЕШЕНИЕ ТВОЕЙ ПРОБЛЕМЫ)
        let people = [];
        let items = [];

        cleanBoxes.forEach(b => {
            let name = labels[b.classId] ? labels[b.classId].toLowerCase() : "";
            if (name.includes("person") || name === "human") {
                people.push(b);
            } else {
                items.push({...b, name: name, used: false});
            }
        });

        let safeCount = 0;
        let badCount = 0;

        // Проверяем каждого найденного человека
        people.forEach(p => {
            let isSafe = false;
            let isDanger = false;
            let pCenterX = p.x + p.w/2;
            let pCenterY = p.y + p.h/2;

            // Ищем каску или жилет внутри этого человека
            items.forEach(item => {
                let iCenterX = item.x + item.w/2;
                let iCenterY = item.y + item.h/2;

                // Если центр СИЗ находится внутри рамки человека
                if (iCenterX > p.x && iCenterX < p.x + p.w && iCenterY > p.y && iCenterY < p.y + p.h) {
                    item.used = true; // Отмечаем, что этот предмет принадлежит этому человеку
                    if (item.name.includes("no-")) isDanger = true;
                    else if (item.name.includes("helmet") || item.name.includes("vest")) isSafe = true;
                }
            });

            // Выносим вердикт: ОДИН человек = ОДНА рамка
            let status = "";
            let color = "";
            
            if (isDanger) {
                status = "БЕЗ СИЗ"; color = "#ef4444"; badCount++;
            } else if (isSafe) {
                status = "В ЗАЩИТЕ"; color = "#22c55e"; safeCount++;
            } else {
                // Если нейросеть вообще не нашла на нем ни каски, ни "no-helmet"
                status = "БЕЗ СИЗ"; color = "#ef4444"; badCount++;
            }

            // Рисуем одну финальную рамку на человека
            ctx.strokeStyle = color; ctx.lineWidth = 4;
            ctx.strokeRect(p.x, p.y, p.w, p.h);
            ctx.fillStyle = color; ctx.fillRect(p.x, p.y > 25 ? p.y - 25 : 0, p.w + 20, 25);
            ctx.fillStyle = "white"; ctx.font = "bold 14px Arial";
            ctx.fillText(`${status} (${Math.round(p.prob*100)}%)`, p.x + 5, p.y > 25 ? p.y - 7 : 18);
        });

        // Шаг 3: Если каска просто лежит на столе (без человека), рисуем её отдельно
        items.forEach(item => {
            if (!item.used) {
                let isDanger = item.name.includes("no-");
                let color = isDanger ? "#ef4444" : "#22c55e";
                let text = isDanger ? "НАРУШЕНИЕ" : "СИЗ";
                if (isDanger) badCount++; else safeCount++;

                ctx.strokeStyle = color; ctx.lineWidth = 3;
                ctx.strokeRect(item.x, item.y, item.w, item.h);
                ctx.fillStyle = color; ctx.font = "bold 12px Arial";
                ctx.fillText(`${text} ${Math.round(item.prob*100)}%`, item.x, item.y - 5);
            }
        });

        // Обновляем статистику
        document.getElementById('count-safe').innerText = safeCount;
        document.getElementById('count-bad').innerText = badCount;
    }

    function iou(b1, b2) {
        let x1 = Math.max(b1.x, b2.x), y1 = Math.max(b1.y, b2.y);
        let x2 = Math.min(b1.x + b1.w, b2.x + b2.w), y2 = Math.min(b1.y + b1.h, b2.y + b2.h);
        let inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        let union = b1.w * b1.h + b2.w * b2.h - inter;
        return inter / union;
    }

    confRange.addEventListener('input', (e) => {
        confValLabel.innerText = e.target.value + '%';
        // Если фото уже загружено, перерисовываем его с новой чувствительностью сразу
        if (!isLive && document.getElementById('file-input').files[0]) {
            runInference(canvas); 
        }
    });

    init();
</script>
</body>
</html>
