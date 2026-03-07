<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>SafeGuard | PPE Inspector</title>
  <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
  <style>
    :root{--bg:#0f172a;--card:#1e293b;--p:#3b82f6;--ok:#22c55e;--bad:#ef4444;--mut:#94a3b8;}
    *{box-sizing:border-box}
    body{margin:0;background:var(--bg);color:#fff;font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;padding:16px}
    .wrap{max-width:1100px;margin:0 auto;display:grid;grid-template-columns:340px 1fr;gap:16px}
    .panel,.main{background:var(--card);border-radius:16px;box-shadow:0 15px 35px rgba(0,0,0,.35)}
    .panel{padding:14px}
    .main{padding:14px}
    h1{margin:0 0 8px;font-size:20px;color:var(--p)}
    .small{font-size:12px;color:var(--mut);line-height:1.35}
    label{display:block;font-size:12px;color:var(--mut);margin-top:10px;margin-bottom:6px}
    input[type="text"]{width:100%;padding:10px;border-radius:10px;border:1px solid #334155;background:#0b1220;color:#fff}
    input[type="range"]{width:100%}
    .row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
    .btns{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:12px}
    button{padding:12px;border-radius:12px;border:0;background:var(--p);color:#fff;font-weight:700;cursor:pointer}
    button.secondary{background:#8b5cf6}
    .status{margin-top:10px;font-size:12px;color:#38bdf8;word-break:break-word}
    .monitor{position:relative;width:100%;aspect-ratio:4/3;background:#000;border:2px solid #334155;border-radius:14px;overflow:hidden}
    canvas, video{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}
    video{display:none}
    .stats{margin-top:12px;display:grid;grid-template-columns:1fr 1fr;gap:10px}
    .stat{background:#0b1220;border:1px solid #334155;border-radius:14px;padding:12px;text-align:center}
    .stat .v{font-size:28px;font-weight:800}
    .stat .t{font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.08em}
    .ok{color:var(--ok)} .bad{color:var(--bad)}
    .hint{margin-top:10px;font-size:12px;color:var(--mut)}
    input[type=file]{display:none}
    .filelabel{display:block;text-align:center;background:#8b5cf6;padding:12px;border-radius:12px;font-weight:700;cursor:pointer}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="panel">
      <h1>SafeGuard PRO</h1>
      <div class="small">
        Рисуем <b>ОДНУ рамку на человека</b>. СИЗ используются только для определения статуса.<br/>
        Нужно один раз указать ID классов из твоей модели.
      </div>

      <label>Confidence (порог уверенности): <span id="thrText">35%</span></label>
      <input id="thr" type="range" min="5" max="95" step="5" value="35" />

      <div class="row">
        <div>
          <label>ID класса(ов) PERSON (через запятую)</label>
          <input id="personIds" type="text" value="3" placeholder="например: 0 или 3" />
        </div>
        <div>
          <label>ID класса(ов) Helmet+Vest (через запятую)</label>
          <input id="ppeIds" type="text" value="0,4" placeholder="например: 0,4" />
        </div>
      </div>

      <div class="row">
        <div>
          <label>ID класса(ов) NO-Helmet/NO-Vest (через запятую)</label>
          <input id="noPpeIds" type="text" value="1,2" placeholder="например: 1,2" />
        </div>
        <div>
          <label>FPS лимит (для видео)</label>
          <input id="fps" type="text" value="10" />
        </div>
      </div>

      <div class="btns">
        <button id="btnCam">Камера</button>
        <label class="filelabel">
          Фото
          <input id="file" type="file" accept="image/*" />
        </label>
      </div>

      <div class="status" id="status">Загрузка best.onnx...</div>

      <div class="hint">
        Если снова “наслаивает”, значит неверные ID. Поменяй personIds/ppeIds/noPpeIds.
      </div>
    </div>

    <div class="main">
      <div class="monitor">
        <video id="video" autoplay muted playsinline></video>
        <canvas id="cv"></canvas>
      </div>

      <div class="stats">
        <div class="stat">
          <div class="v ok" id="safe">0</div>
          <div class="t">в защите</div>
        </div>
        <div class="stat">
          <div class="v bad" id="bad">0</div>
          <div class="t">без сиз</div>
        </div>
      </div>
    </div>
  </div>

<script>
  const statusEl = document.getElementById('status');
  const thrEl = document.getElementById('thr');
  const thrText = document.getElementById('thrText');
  const personIdsEl = document.getElementById('personIds');
  const ppeIdsEl = document.getElementById('ppeIds');
  const noPpeIdsEl = document.getElementById('noPpeIds');
  const fpsEl = document.getElementById('fps');

  const btnCam = document.getElementById('btnCam');
  const fileEl = document.getElementById('file');

  const video = document.getElementById('video');
  const canvas = document.getElementById('cv');
  const ctx = canvas.getContext('2d');

  const safeEl = document.getElementById('safe');
  const badEl = document.getElementById('bad');

  let session = null;
  let live = false;
  let lastFrameTime = 0;

  function parseIds(str){
    return str.split(',').map(s => s.trim()).filter(s => s.length).map(s => Number(s)).filter(n => Number.isFinite(n));
  }

  function sigmoid(x){ return 1 / (1 + Math.exp(-x)); }

  // IoU for NMS
  function iou(a,b){
    const x1 = Math.max(a.x1,b.x1), y1 = Math.max(a.y1,b.y1);
    const x2 = Math.min(a.x2,b.x2), y2 = Math.min(a.y2,b.y2);
    const inter = Math.max(0,x2-x1)*Math.max(0,y2-y1);
    const areaA = Math.max(0,a.x2-a.x1)*Math.max(0,a.y2-a.y1);
    const areaB = Math.max(0,b.x2-b.x1)*Math.max(0,b.y2-b.y1);
    const uni = areaA + areaB - inter;
    return uni <= 0 ? 0 : inter/uni;
  }

  function nms(boxes, iouThr){
    boxes.sort((a,b)=>b.score-a.score);
    const out=[];
    while(boxes.length){
      const best = boxes.shift();
      out.push(best);
      boxes = boxes.filter(b => iou(best,b) < iouThr);
    }
    return out;
  }

  async function loadModel(){
    try{
      statusEl.textContent = "Загрузка best.onnx (локально из GitHub Pages)...";
      session = await ort.InferenceSession.create('./best.onnx', {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
      });
      statusEl.textContent = "Модель загружена. Можно запускать.";
    }catch(e){
      console.error(e);
      statusEl.textContent = "Ошибка загрузки best.onnx. Проверь, что файл лежит рядом с index.html и GitHub Pages включен.";
    }
  }

  function toTensorFromSource(source, size=640){
    const off = document.createElement('canvas');
    off.width=size; off.height=size;
    const octx = off.getContext('2d');
    octx.drawImage(source,0,0,size,size);
    const img = octx.getImageData(0,0,size,size).data;

    const chw = new Float32Array(3*size*size);
    const area = size*size;
    for(let i=0;i<area;i++){
      chw[i] = img[i*4]/255;
      chw[i+area] = img[i*4+1]/255;
      chw[i+2*area] = img[i*4+2]/255;
    }
    return new ort.Tensor('float32', chw, [1,3,size,size]);
  }

  // универсальный разбор выхода YOLOv8: [1, C, A] или [1, A, C]
  function decode(outputTensor, thr){
    const data = outputTensor.data;
    const dims = outputTensor.dims; // ex: [1, 9, 8400] or [1, 8400, 9]

    const A = (dims[1] === 8400) ? dims[1] : dims[2];
    const C = (dims[1] === 8400) ? dims[2] : dims[1]; // channels
    const transposed = (dims[1] === 8400); // [1, A, C]

    const numClasses = C - 4;
    const dets = [];

    // NOTE: многие onnx выходы уже "sigmoid", но некоторые - logits.
    // сделаем авто: если видим значения > 1.5, применим sigmoid к score.
    let sampleMax = 0;
    for(let k=0;k<Math.min(2000, data.length);k++){
      const v = Math.abs(data[k]);
      if(v>sampleMax) sampleMax=v;
    }
    const needSigmoid = sampleMax > 1.5;

    for(let i=0;i<A;i++){
      let cx,cy,w,h;
      if(transposed){
        const base = i*C;
        cx=data[base+0]; cy=data[base+1]; w=data[base+2]; h=data[base+3];
        // class scores
        let bestScore=-1, bestId=-1;
        for(let c=0;c<numClasses;c++){
          let s = data[base+4+c];
          if(needSigmoid) s = sigmoid(s);
          if(s>bestScore){bestScore=s; bestId=c;}
        }
        if(bestScore>=thr){
          dets.push({cx,cy,w,h,score:bestScore,cls:bestId});
        }
      } else {
        // [1, C, A] packed by channel
        cx=data[0*A+i]; cy=data[1*A+i]; w=data[2*A+i]; h=data[3*A+i];
        let bestScore=-1, bestId=-1;
        for(let c=0;c<numClasses;c++){
          let s = data[(4+c)*A + i];
          if(needSigmoid) s = sigmoid(s);
          if(s>bestScore){bestScore=s; bestId=c;}
        }
        if(bestScore>=thr){
          dets.push({cx,cy,w,h,score:bestScore,cls:bestId});
        }
      }
    }
    return dets;
  }

  function drawPeopleStatus(source, dets){
    // отрисовка: только PERSON рамки, статус по PPE
    canvas.width = source.videoWidth || source.width;
    canvas.height = source.videoHeight || source.height;
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.drawImage(source,0,0,canvas.width,canvas.height);

    const personIds = parseIds(personIdsEl.value);
    const ppeIds = parseIds(ppeIdsEl.value);
    const noPpeIds = parseIds(noPpeIdsEl.value);

    // scale from 640 model space -> canvas space
    const sx = canvas.width / 640;
    const sy = canvas.height / 640;

    // convert to xyxy boxes
    const all = dets.map(d => {
      const x1 = (d.cx - d.w/2)*sx;
      const y1 = (d.cy - d.h/2)*sy;
      const x2 = (d.cx + d.w/2)*sx;
      const y2 = (d.cy + d.h/2)*sy;
      return {x1,y1,x2,y2,score:d.score,cls:d.cls};
    });

    // separate by class groups
    const personsRaw = all.filter(b => personIds.includes(b.cls));
    const ppeRaw = all.filter(b => ppeIds.includes(b.cls));
    const noPpeRaw = all.filter(b => noPpeIds.includes(b.cls));

    // NMS only for persons to guarantee 1 box per person
    const persons = nms(personsRaw, 0.45);

    let safe=0, bad=0;

    for(const p of persons){
      // status: if any NO-PPE inside person => bad
      // else if any PPE inside => safe
      // else => bad (нет данных о защите)
      const inside = (b) => {
        const cx = (b.x1+b.x2)/2;
        const cy = (b.y1+b.y2)/2;
        return (cx>p.x1 && cx<p.x2 && cy>p.y1 && cy<p.y2);
      };

      const hasNo = noPpeRaw.some(inside);
      const hasPpe = ppeRaw.some(inside);

      const isSafe = (!hasNo && hasPpe);
      if(isSafe) safe++; else bad++;

      const color = isSafe ? getComputedStyle(document.documentElement).getPropertyValue('--ok').trim()
                           : getComputedStyle(document.documentElement).getPropertyValue('--bad').trim();
      const text = isSafe ? "В ЗАЩИТЕ" : "БЕЗ СИЗ";

      // draw person box ONCE
      ctx.strokeStyle = color;
      ctx.lineWidth = 4;
      ctx.strokeRect(p.x1, p.y1, p.x2-p.x1, p.y2-p.y1);

      // head highlight if bad
      if(!isSafe){
        const headH = (p.y2-p.y1)*0.25;
        ctx.fillStyle = "rgba(239,68,68,0.25)";
        ctx.fillRect(p.x1, p.y1, p.x2-p.x1, headH);
      }

      // label
      ctx.fillStyle = color;
      ctx.fillRect(p.x1, Math.max(0,p.y1-26), Math.min(260, (p.x2-p.x1)), 26);
      ctx.fillStyle = "#fff";
      ctx.font = "bold 14px system-ui";
      ctx.fillText(text, p.x1+6, Math.max(18,p.y1-8));
    }

    safeEl.textContent = safe;
    badEl.textContent = bad;
  }

  async function run(source){
    if(!session) return;
    const thr = Number(thrEl.value)/100;
    const input = toTensorFromSource(source, 640);
    const feeds = {[session.inputNames[0]]: input};
    const out = await session.run(feeds);
    const outTensor = out[session.outputNames[0]];
    const dets = decode(outTensor, thr);
    drawPeopleStatus(source, dets);
  }

  // UI: threshold text
  const thrEl = document.getElementById('thr');
  thrEl.addEventListener('input', ()=>{
    thrText.textContent = `${thrEl.value}%`;
  });
  // init threshold text
  const thrText = document.getElementById('thrText');
  thrText.textContent = `${thrEl.value}%`;

  // camera
  btnCam.addEventListener('click', async ()=>{
    if(!session) return;
    const stream = await navigator.mediaDevices.getUserMedia({video:{facingMode:"environment"}});
    video.srcObject = stream;
    video.style.display = "block";
    live = true;
    video.onloadedmetadata = ()=>{
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      statusEl.textContent = "Камера активна. ИИ работает.";
      requestAnimationFrame(loop);
    };
  });

  async function loop(ts){
    if(!live) return;
    const fpsLimit = Math.max(1, Math.min(30, Number(fpsEl.value)||10));
    const minDt = 1000 / fpsLimit;
    if(ts - lastFrameTime >= minDt){
      lastFrameTime = ts;
      try{ await run(video); } catch(e){ console.error(e); statusEl.textContent="Ошибка инференса (см. Console)"; }
    }
    requestAnimationFrame(loop);
  }

  // file
  fileEl.addEventListener('change', async (e)=>{
    const f = e.target.files[0];
    if(!f) return;
    live = false;
    if(video.srcObject) video.srcObject.getTracks().forEach(t=>t.stop());
    const img = new Image();
    img.src = URL.createObjectURL(f);
    img.onload = async ()=>{
      statusEl.textContent = "Анализ фото...";
      await run(img);
      statusEl.textContent = "Готово.";
    }
  });

  // show percent on slider
  thrEl.addEventListener('input', ()=>{ thrText.textContent = `${thrEl.value}%`; });

  loadModel();
</script>
</body>
</html>
