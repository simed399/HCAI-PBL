// helpers --------------------------------------------------------------
const $ = (q) => document.querySelector(q);

function makeGrid(container) {
  container.innerHTML = "";
  for (let i = 0; i < 25; i++) {
    const d = document.createElement("div");
    d.className = "cell empty";
    container.appendChild(d);
  }
}

function renderFrame(container, arr2d) {
  const flat = [].concat(...arr2d);
  const cells = container.children;
  for (let i = 0; i < 25; i++) {
    const v = Number(flat[i]); 
    const cls = window.STATIC_MAP[v] || "cell empty";
    cells[i].className = `cell ${cls.split(" ")[1]}`;
    if (v === 1) cells[i].textContent = "🐭";
    else if (v === 2) cells[i].textContent = "🧀";
    else if (v === 5) cells[i].textContent = "🧀";
    else if (v === 3) cells[i].textContent = "💀";
    else if (v === 4) cells[i].textContent = "🧱";
    else cells[i].textContent = "";
  }
}

async function fetchRollout(kind) {
  const res = await fetch(`./rollout/${kind}`);
  const data = await res.json();
  console.debug("rollout", kind, data);
  return data;
}

function playFrames(container, frames, metaEl, pillEl) {
  clearInterval(container._timer);
  if (!frames || !frames.length) {
    if (metaEl) metaEl.textContent = "No frames (train first?)";
    if (pillEl) pillEl.textContent = "0 frames";
    return;
  }
  renderFrame(container, frames[0]);
  if (metaEl) metaEl.textContent = `Frame 1/${frames.length}`;
  if (pillEl) pillEl.textContent = `Steps: ${frames.length}`;
  let i = 1;
  container._timer = setInterval(() => {
    renderFrame(container, frames[i]);
    if (metaEl) metaEl.textContent = `Frame ${i + 1}/${frames.length}`;
    i++;
    if (i >= frames.length) clearInterval(container._timer);
  }, 180);
}

// mini player for the A/B preference grids
function playSequence(container, frames, metaEl, speed = 180) {
  clearInterval(container._timer);
  if (!frames || !frames.length) { if (metaEl) metaEl.textContent = "No frames"; return; }
  renderFrame(container, frames[0]);
  if (metaEl) metaEl.textContent = `Frame 1/${frames.length}`;
  let i = 1;
  container._timer = setInterval(() => {
    renderFrame(container, frames[i]);
    if (metaEl) metaEl.textContent = `Frame ${i + 1}/${frames.length}`;
    i++;
    if (i >= frames.length) { clearInterval(container._timer); if (metaEl) metaEl.textContent = `Done at ${frames.length}`; }
  }, speed);
}

async function updateStatus() {
  const out = $("#statusOut");
  const r = await fetch("./status"); const d = await r.json();
  if (!d.ok) { out.textContent = "Status error"; return; }
  const items = d.artifacts.map(a => a.exists
    ? `<span class="chip ok">✅ ${a.name} — ${a.size} bytes — ${a.mtime} — <a href="${a.download}">download</a></span>`
    : `<span class="chip no">❌ ${a.name} — not trained</span>`).join(" ");
  out.innerHTML = items;
}

(function injectChipsCSS(){
  const style = document.createElement("style");
  style.textContent = `
   .chip{display:inline-block;padding:6px 10px;border:1px solid #e5e7eb;border-radius:999px;margin-right:6px;background:#fff;font-size:13px}
   .chip.ok{border-color:#d1fae5;background:#ecfdf5}
   .chip.no{border-color:#fde68a;background:#fffbeb}
   .chip a{text-decoration:none;font-weight:600}
  `;
  document.head.appendChild(style);
})();

// human feedback (Extra option a fake user is coded bz default) ---------------------------------------------------
let currentPairId = null;

async function fetchNewPair() {
  const r = await fetch("./pref/new");
  const d = await r.json();
  if (!d.ok) { $("#prefCount").textContent = d.error || "Error"; return; }
  currentPairId = d.pair_id;

  if ($("#gridA").children.length === 0) makeGrid($("#gridA"));
  if ($("#gridB").children.length === 0) makeGrid($("#gridB"));

  playSequence($("#gridA"), d.A || [], $("#metaA"));
  playSequence($("#gridB"), d.B || [], $("#metaB"));
}

async function submitChoice(side) {
  if (!currentPairId) { $("#prefCount").textContent = "Get a new pair first."; return; }
  clearInterval($("#gridA")._timer); clearInterval($("#gridB")._timer);
  const r = await fetch("./pref/choose", {
    method: "POST", headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ pair_id: currentPairId, choice: side })
  });
  const d = await r.json();
  if (!d.ok) { $("#prefCount").textContent = d.error || "Error"; return; }
  $("#prefCount").textContent = `Saved. Total labels: ${d.count}`;
  currentPairId = null;
}

// init -----------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  console.log(">>> app.js boot", new Date().toLocaleString());

  // base grids
  if ($("#gridBase") && $("#gridBase").children.length === 0) makeGrid($("#gridBase"));
  if ($("#gridRLHF") && $("#gridRLHF").children.length === 0) makeGrid($("#gridRLHF"));

  updateStatus();

  // training buttons
  $("#trainBtn")?.addEventListener("click", async () => {
    $("#statusOut").textContent = "Training baseline...";
    const r = await fetch("./run-demo", { method:"POST" });
    const d = await r.json();
    $("#statusOut").textContent = d.ok ? d.msg : (d.error || "Error");
    updateStatus();
  });

  $("#trainRewardBtn")?.addEventListener("click", async () => {
    $("#statusOut").textContent = "Training reward model...";
    const useReal = $("#useReal")?.checked || false;
    const r = await fetch("./train-reward", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ use_real: useReal })
    });
    const d = await r.json();
    $("#statusOut").textContent = d.ok ? d.msg : (d.error || "Error");
    updateStatus();
  });

  $("#trainRLHFBtn")?.addEventListener("click", async () => {
    $("#statusOut").textContent = "Training RLHF...";
    const r = await fetch("./run-rlhf", { method:"POST" });
    const d = await r.json();
    $("#statusOut").textContent = d.ok ? d.msg : (d.error || "Error");
    updateStatus();
  });

  $("#checkStatusBtn")?.addEventListener("click", updateStatus);

  // play + compare
  $("#playBase")?.addEventListener("click", async () => {
    const pill = $("#pillBase"); if (pill) pill.textContent = "…";
    const d = await fetchRollout("base");
    if (!d.ok) { if (pill) pill.textContent = "error"; return; }
    if (pill) pill.textContent = `Return: ${d.reward_sum.toFixed(1)}`;
    playFrames($("#gridBase"), d.frames, $("#metaBase"), pill);
  });

  $("#playRLHF")?.addEventListener("click", async () => {
    const pill = $("#pillRLHF"); if (pill) pill.textContent = "…";
    const d = await fetchRollout("rlhf");
    if (!d.ok) { if (pill) pill.textContent = "error"; return; }
    if (pill) pill.textContent = `Return: ${d.reward_sum.toFixed(1)}`;
    playFrames($("#gridRLHF"), d.frames, $("#metaRLHF"), pill);
  });

  $("#compareBtn")?.addEventListener("click", async () => {
    const [b, r] = await Promise.all([fetchRollout("base"), fetchRollout("rlhf")]);
    if (!b.ok || !r.ok) { $("#statusOut").textContent = "Play error"; return; }
    $("#pillBase").textContent = `Return: ${b.reward_sum.toFixed(1)}`;
    $("#pillRLHF").textContent = `Return: ${r.reward_sum.toFixed(1)}`;
    const frames = Math.min(b.frames.length, r.frames.length);
    let i = 0;
    const timer = setInterval(() => {
      renderFrame($("#gridBase"), b.frames[i]);
      renderFrame($("#gridRLHF"), r.frames[i]);
      $("#metaBase").textContent = `Frame ${i + 1}/${b.frames.length}`;
      $("#metaRLHF").textContent = `Frame ${i + 1}/${r.frames.length}`;
      i++; if (i >= frames) clearInterval(timer);
    }, 180);
  });

  // human feedback
  $("#newPair")?.addEventListener("click", fetchNewPair);
  $("#chooseA")?.addEventListener("click", () => submitChoice("A"));
  $("#chooseB")?.addEventListener("click", () => submitChoice("B"));
});
