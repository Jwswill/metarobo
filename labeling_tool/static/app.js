const CLASS_NAMES = ["safety", "danger", "corn"];
const CLASS_COLORS = ["#2ecc71", "#e74c3c", "#f39c12"];

const dirSelect = document.getElementById("dirSelect");
const refreshDirsBtn = document.getElementById("refreshDirs");
const progressText = document.getElementById("progressText");
const imageListEl = document.getElementById("imageList");
const boxListEl = document.getElementById("boxList");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const fileLabel = document.getElementById("fileLabel");
const statusMsg = document.getElementById("statusMsg");
const prevBtn = document.getElementById("prevBtn");
const nextBtn = document.getElementById("nextBtn");
const saveBtn = document.getElementById("saveBtn");
const rotateCwBtn = document.getElementById("rotateCwBtn");
const rotateCcwBtn = document.getElementById("rotateCcwBtn");
const classBtns = Array.from(document.querySelectorAll(".class-btn"));

let state = {
  dir: null,
  images: [],       // [{name, labeled, box_count}]
  index: -1,
  img: new Image(),
  naturalW: 0,
  naturalH: 0,
  boxes: [],         // [{class_id, x1, y1, x2, y2}] in natural pixel coords
  selected: -1,
  currentClass: 0,
  dirty: false,
  drawing: null,     // {x1,y1,x2,y2} while dragging
};

function setStatus(msg, timeout = 1500) {
  statusMsg.textContent = msg;
  if (timeout) setTimeout(() => { if (statusMsg.textContent === msg) statusMsg.textContent = ""; }, timeout);
}

async function loadDirs(selectName) {
  const res = await fetch("/api/dirs");
  const data = await res.json();
  dirSelect.innerHTML = "";
  for (const d of data.dirs) {
    const opt = document.createElement("option");
    opt.value = d.name;
    opt.textContent = `${d.name} (${d.labeled}/${d.total})`;
    dirSelect.appendChild(opt);
  }
  if (data.dirs.length === 0) {
    setStatus("cap_image 디렉터리에 프레임이 없습니다.", 5000);
    return;
  }
  const target = selectName && data.dirs.some(d => d.name === selectName) ? selectName : data.dirs[0].name;
  dirSelect.value = target;
  await loadImages(target);
}

async function loadImages(dirName) {
  state.dir = dirName;
  const res = await fetch(`/api/images?dir=${encodeURIComponent(dirName)}`);
  const data = await res.json();
  state.images = data.images;
  renderImageList();
  updateProgress();
  const firstUnlabeled = state.images.findIndex(im => !im.labeled);
  const idx = firstUnlabeled >= 0 ? firstUnlabeled : 0;
  await loadImage(idx);
}

function updateProgress() {
  const labeled = state.images.filter(im => im.labeled).length;
  progressText.textContent = state.images.length
    ? `${state.dir}: ${labeled}/${state.images.length} 라벨링 완료`
    : "";
}

function renderImageList() {
  imageListEl.innerHTML = "";
  state.images.forEach((im, i) => {
    const li = document.createElement("li");
    li.className = i === state.index ? "current" : (im.labeled ? "" : "empty");
    li.textContent = im.name;
    const badge = document.createElement("span");
    badge.className = "badge";
    badge.textContent = im.labeled ? `✓${im.box_count}` : "";
    li.appendChild(badge);
    li.addEventListener("click", () => loadImage(i));
    imageListEl.appendChild(li);
  });
}

async function saveCurrent(force = false) {
  if (state.index < 0 || (!state.dirty && !force)) return;
  const image = state.images[state.index];
  const annotations = state.boxes.map(b => {
    const x1 = Math.min(b.x1, b.x2), x2 = Math.max(b.x1, b.x2);
    const y1 = Math.min(b.y1, b.y2), y2 = Math.max(b.y1, b.y2);
    return {
      class_id: b.class_id,
      class_name: CLASS_NAMES[b.class_id],
      bbox: {
        x_center: (x1 + x2) / 2 / state.naturalW,
        y_center: (y1 + y2) / 2 / state.naturalH,
        width: (x2 - x1) / state.naturalW,
        height: (y2 - y1) / state.naturalH,
      },
    };
  });
  const payload = {
    dir: state.dir,
    image: image.name,
    width: state.naturalW,
    height: state.naturalH,
    annotations,
  };
  const res = await fetch("/api/annotation", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (res.ok) {
    image.labeled = annotations.length > 0 || true; // 저장 파일이 생성됨(빈 배열도 유효한 라벨)
    image.box_count = annotations.length;
    state.dirty = false;
    renderImageList();
    updateProgress();
    setStatus("저장됨");
  } else {
    setStatus("저장 실패", 3000);
  }
}

async function loadImage(idx) {
  if (idx < 0 || idx >= state.images.length) return;
  if (state.dirty) await saveCurrent();

  state.index = idx;
  state.selected = -1;
  const image = state.images[idx];
  fileLabel.textContent = `${idx + 1} / ${state.images.length} — ${image.name}`;
  renderImageList();

  await new Promise((resolve) => {
    state.img = new Image();
    state.img.onload = resolve;
    state.img.src = `/api/image?dir=${encodeURIComponent(state.dir)}&name=${encodeURIComponent(image.name)}&t=${Date.now()}`;
  });
  state.naturalW = state.img.naturalWidth;
  state.naturalH = state.img.naturalHeight;
  canvas.width = state.naturalW;
  canvas.height = state.naturalH;

  const annRes = await fetch(`/api/annotation?dir=${encodeURIComponent(state.dir)}&name=${encodeURIComponent(image.name)}`);
  const ann = await annRes.json();
  state.boxes = (ann.annotations || []).map(a => {
    const w = state.naturalW, h = state.naturalH;
    const cx = a.bbox.x_center * w, cy = a.bbox.y_center * h;
    const bw = a.bbox.width * w, bh = a.bbox.height * h;
    return { class_id: a.class_id, x1: cx - bw / 2, y1: cy - bh / 2, x2: cx + bw / 2, y2: cy + bh / 2 };
  });
  state.dirty = false;
  renderBoxList();
  draw();
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(state.img, 0, 0);

  state.boxes.forEach((b, i) => {
    const color = CLASS_COLORS[b.class_id];
    ctx.strokeStyle = color;
    ctx.lineWidth = i === state.selected ? 3 : 2;
    const x = Math.min(b.x1, b.x2), y = Math.min(b.y1, b.y2);
    const w = Math.abs(b.x2 - b.x1), h = Math.abs(b.y2 - b.y1);
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = color;
    ctx.font = "12px sans-serif";
    const label = CLASS_NAMES[b.class_id];
    const tw = ctx.measureText(label).width + 6;
    ctx.fillRect(x, Math.max(0, y - 14), tw, 14);
    ctx.fillStyle = "#111";
    ctx.fillText(label, x + 3, Math.max(10, y - 3));
  });

  if (state.drawing) {
    const d = state.drawing;
    ctx.strokeStyle = CLASS_COLORS[state.currentClass];
    ctx.setLineDash([4, 3]);
    ctx.lineWidth = 2;
    const x = Math.min(d.x1, d.x2), y = Math.min(d.y1, d.y2);
    const w = Math.abs(d.x2 - d.x1), h = Math.abs(d.y2 - d.y1);
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);
  }
}

function renderBoxList() {
  boxListEl.innerHTML = "";
  state.boxes.forEach((b, i) => {
    const li = document.createElement("li");
    li.className = i === state.selected ? "selected" : "";
    const sw = document.createElement("span");
    sw.className = "swatch";
    sw.style.background = CLASS_COLORS[b.class_id];
    li.appendChild(sw);
    const label = document.createElement("span");
    label.textContent = `${CLASS_NAMES[b.class_id]}`;
    li.appendChild(label);
    const del = document.createElement("button");
    del.textContent = "삭제";
    del.addEventListener("click", (e) => { e.stopPropagation(); removeBox(i); });
    li.appendChild(del);
    li.addEventListener("click", () => { state.selected = i; renderBoxList(); draw(); });
    boxListEl.appendChild(li);
  });
}

function removeBox(i) {
  state.boxes.splice(i, 1);
  state.selected = -1;
  state.dirty = true;
  renderBoxList();
  draw();
}

function getCanvasPos(evt) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  let x = (evt.clientX - rect.left) * scaleX;
  let y = (evt.clientY - rect.top) * scaleY;
  x = Math.max(0, Math.min(canvas.width, x));
  y = Math.max(0, Math.min(canvas.height, y));
  return { x, y };
}

canvas.addEventListener("mousedown", (evt) => {
  const p = getCanvasPos(evt);
  state.drawing = { x1: p.x, y1: p.y, x2: p.x, y2: p.y };
});

canvas.addEventListener("mousemove", (evt) => {
  if (!state.drawing) return;
  const p = getCanvasPos(evt);
  state.drawing.x2 = p.x;
  state.drawing.y2 = p.y;
  draw();
});

window.addEventListener("mouseup", () => {
  if (!state.drawing) return;
  const d = state.drawing;
  state.drawing = null;
  const w = Math.abs(d.x2 - d.x1), h = Math.abs(d.y2 - d.y1);
  if (w >= 3 && h >= 3) {
    state.boxes.push({ class_id: state.currentClass, x1: d.x1, y1: d.y1, x2: d.x2, y2: d.y2 });
    state.selected = state.boxes.length - 1;
    state.dirty = true;
    renderBoxList();
  }
  draw();
});

async function rotateCurrent(direction) {
  if (state.index < 0) return;
  if (state.dirty) {
    const ok = confirm("저장하지 않은 변경사항이 있습니다. 회전 전에 저장할까요?");
    if (ok) await saveCurrent(true);
  }
  const image = state.images[state.index];
  setStatus("회전 중...", 0);
  try {
    const res = await fetch("/api/rotate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dir: state.dir, image: image.name, direction }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      setStatus(`회전 실패: ${err.error || res.status}`, 3000);
      return;
    }
    setStatus("회전 완료 (원본 파일 덮어씀)");
    await loadImage(state.index); // 회전된 이미지 + 변환된 라벨을 서버에서 다시 불러옴
  } catch (e) {
    setStatus("회전 실패", 3000);
  }
}

function setCurrentClass(idx) {
  state.currentClass = idx;
  classBtns.forEach((b, i) => b.classList.toggle("active", i === idx));
  if (state.selected >= 0) {
    state.boxes[state.selected].class_id = idx;
    state.dirty = true;
    renderBoxList();
    draw();
  }
}

classBtns.forEach((btn, i) => btn.addEventListener("click", () => setCurrentClass(i)));
setCurrentClass(0);

prevBtn.addEventListener("click", () => loadImage(state.index - 1));
nextBtn.addEventListener("click", () => loadImage(state.index + 1));
saveBtn.addEventListener("click", () => saveCurrent(true));
rotateCwBtn.addEventListener("click", () => rotateCurrent("cw"));
rotateCcwBtn.addEventListener("click", () => rotateCurrent("ccw"));
dirSelect.addEventListener("change", () => loadImages(dirSelect.value));
refreshDirsBtn.addEventListener("click", () => loadDirs(dirSelect.value));

window.addEventListener("keydown", (evt) => {
  const tag = document.activeElement.tagName;
  if (tag === "SELECT" || tag === "INPUT" || tag === "TEXTAREA") return;

  if (evt.key === "1") setCurrentClass(0);
  else if (evt.key === "2") setCurrentClass(1);
  else if (evt.key === "3") setCurrentClass(2);
  else if (evt.key === "a" || evt.key === "A" || evt.key === "ArrowLeft") loadImage(state.index - 1);
  else if (evt.key === "d" || evt.key === "D" || evt.key === "ArrowRight") loadImage(state.index + 1);
  else if (evt.key === "s" || evt.key === "S") { evt.preventDefault(); saveCurrent(true); }
  else if (evt.key === "R") { evt.preventDefault(); rotateCurrent("ccw"); }
  else if (evt.key === "r") { evt.preventDefault(); rotateCurrent("cw"); }
  else if (evt.key === "Delete" || evt.key === "Backspace") {
    if (state.selected >= 0) { evt.preventDefault(); removeBox(state.selected); }
  }
});

loadDirs();
