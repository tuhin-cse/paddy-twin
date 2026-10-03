// Paddy Twin front end: renders one simulated season from /api/season and asks farmers via /api/ask.

const $ = (s) => document.querySelector(s);
const SLIDERS = [
  ["n_fields", "Fields in the village", 12, (v) => `${v}`, "two map rows per tubewell"],
  ["leak", "Soil leakiness (S&P, mm/d)", 0.5, (v) => v.toFixed(1), "median base seepage + percolation"],
  ["et0_scale", "Evaporation (ET₀ ×)", 0.05, (v) => `×${v.toFixed(2)}`, "hotter season → higher"],
  ["rain_scale", "Rain", 0.1, (v) => `×${v.toFixed(1)}`, ""],
  ["cloud", "Extra cloud", 0.05, pct, "share of Sentinel-2 passes lost"],
  ["share_logging", "Farmers using the app", 0.05, pct, "others never log irrigations"],
  ["p_awd", "Farmers doing AWD", 0.05, pct, "irrigate when 150 mm below surface"],
  ["p_topup", "Farmers topping up", 0.05, pct, "irrigate before the field dries"],
  ["pump_bias", "Pump rating error", 0.05, (v) => `×${v.toFixed(2)}`, "logged depth ÷ true depth"],
  ["nisar_acc", "NISAR accuracy (simulated)", 0.05, pct, "how often a NISAR wet/dry call is right"],
  ["nisar_assumed", "NISAR accuracy the app assumes", 0.05, pct, "the weight a pass gets in the update"],
];
function pct(v) { return `${Math.round(v * 100)} %`; }

let R = null;          // current season
let DEFS = null;       // /api/defaults: defaults and limits
let day = 30, view = "3d", sel = null, timer = null, seed = 1;
let village = null, loading3D = false;   // 3D scene (scene3d.js), loaded on first use
let passIdx = new Map(), asks = new Map(), layout = null, colors = {};
// satellites: Sentinel-2 (optical, 10:30), Sentinel-1 (C-band SAR, 18:00), NISAR (L-band SAR, 06:00)
const SENSORS = { nisar: { label: "NISAR", color: "#8b5cf6" }, s2: { label: "Sentinel-2", color: "#1b9fc6" }, s1: { label: "Sentinel-1", color: "#e07b2c" } };
const SENSOR_ORDER = ["nisar", "s2", "s1"];   // time of day
const passesAt = (f, d) => passIdx.get(f * R.T + d) || [];
const isWrong = (p) => p.truth !== null && p.truth !== undefined && p.wet !== p.truth;
const planted = (f, d) => !!R.planted[f][d];
const replay = () => R && R.mode === "replay";
// data quality (replay): severities and which checks belong to loggers, farmers and the network (app/qc.py)
const SEV = { high: "High", medium: "Medium", low: "Low", info: "Note" };
const QC_GROUPS = { loggers: ["no_data", "stopped", "gap", "late_start", "out_of_range", "flatline", "spike", "drift", "sat_disagree"],
  farmers: ["unlogged", "stopped_logging", "no_app"], network: ["blind", "outage", "meter"] };
let qcFilter = "open", qcDrawn = null;
const qcStatus = (lid) => replay() && R.qc ? (R.qc.loggers.find((l) => l.id === lid) || {}).status : null;
const qcOnDay = (lid, d) => (replay() && R.qc ? R.qc.issues.filter((i) => i.logger === lid && i.start <= d && d <= i.end) : []);
// the advice of day d is for the morning of d + 1; a NISAR pass that morning (06:00) updates it (Experiment 24)
const nisarNext = (f, d) => passesAt(f, d + 1).find((p) => p.s === "nisar" && p.update);
const nisarNote = (f, d) => { const p = nisarNext(f, d); return p ? `<br><span style="color:${SENSORS.nisar.color}">Advice includes the next morning's NISAR pass (said ${p.wet ? "wet" : "dry"})</span>` : ""; };

// ── controls ───────────────────────────────────────────────────────────────
async function init() {
  const d = await (await fetch("/api/defaults")).json();
  DEFS = d;
  const box = $("#sliders");
  for (const [k, label, step, fmt, hint] of SLIDERS) {
    const [lo, hi] = d.limits[k];
    const el = document.createElement("label");
    el.className = "slider";
    el.innerHTML = `<span>${label}</span><span class="v" id="v_${k}"></span>
      <input type="range" id="s_${k}" min="${lo}" max="${hi}" step="${step}" value="${d.defaults[k]}">
      ${hint ? `<span class="hint">${hint}</span>` : ""}`;
    box.appendChild(el);
    const inp = el.querySelector("input");
    const show = () => ($(`#v_${k}`).textContent = fmt(+inp.value));
    inp.addEventListener("input", show);
    show();
  }
  seed = d.defaults.seed;
  const camps = await (await fetch("/api/campaigns")).json();
  $("#source").innerHTML = `<option value="sim">Simulation (simulator v2)</option>` +
    camps.map((c) => `<option value="${c.name}">Campaign: ${c.label}</option>`).join("");
  $("#source").onchange = () => { sel = null; run(); };
  document.querySelectorAll("#sliders .slider").forEach((el) => { if (!el.querySelector("#s_nisar_assumed")) el.classList.add("sim-only"); });
  const sites = await (await fetch("/api/sites")).json();
  $("#site").innerHTML = sites.map((st) => `<option value="${st.name}">${st.label}${st.n_fields ? ` · ${st.n_fields} fields, ${st.n_loggers} loggers` : ""}</option>`).join("");
  const syncSite = () => {
    const grid = $("#site").value === "grid";
    $("#s_n_fields").disabled = !grid;
    $("#v_n_fields").textContent = grid ? $("#s_n_fields").value : "set by the site";
  };
  $("#site").onchange = () => { syncSite(); run(); };
  $("#nisar_update").onchange = () => run();
  $("#model").onchange = () => run();
  syncSite();
  $("#run").onclick = () => run();
  $("#reseed").onclick = () => { seed += 1; run(); };
  $("#day").oninput = (e) => setDay(+e.target.value);
  $("#play").onclick = togglePlay;
  $("#speed").onchange = () => { if (timer) { togglePlay(); togglePlay(); } };
  document.querySelectorAll(".seg button").forEach((b) => (b.onclick = () => {
    document.querySelectorAll(".seg button").forEach((x) => x.classList.toggle("on", x === b));
    view = b.dataset.view; draw(); if (R) legend();
  }));
  document.querySelectorAll("#ask button").forEach((b) => (b.onclick = () => ask(b.dataset.kind)));
  document.querySelectorAll("#dtabs button").forEach((b) => (b.onclick = () => {
    PH.tab = b.dataset.t; document.querySelectorAll("#dtabs button").forEach((x) => x.classList.toggle("on", x === b));
    $("#ask").hidden = replay() || PH.tab === "phone"; phoneShow();
  }));
  const cv = $("#map");
  cv.addEventListener("mousemove", hoverMap);
  cv.addEventListener("mouseleave", () => ($("#tip").hidden = true));
  cv.addEventListener("click", (e) => { const f = hitField(e); if (f !== null) (view === "plan" ? planClick(f) : select(f)); });
  window.addEventListener("resize", () => { if (R) { draw(); drawChart(); } });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { readColors(); draw(); drawChart(); });
  readColors();
  run();
}

function params() {
  const p = { world: $("#world").value, seed, site: $("#site").value || "grid", nisar_update: $("#nisar_update").checked, model: $("#model").value };
  for (const [k] of SLIDERS) p[k] = +$(`#s_${k}`).value;
  return p;
}

let runSeq = 0;
async function run() {
  stop();
  const seqNo = ++runSeq;   // a newer run (a setting changed meanwhile) wins; older results are dropped
  $("#run").disabled = $("#reseed").disabled = true;
  const src = $("#source").value;
  $("#status").textContent = src === "sim" ? "Simulating the season and running the model day by day…" : "Replaying the campaign…";
  const ld = loaderStart(src === "sim" ? "Running the season" : "Replaying the campaign", src === "sim" ? LD_SEASON : LD_REPLAY);
  let next;
  try {
    const body = src === "sim" ? { ...params(), _token: ld.token }
      : { campaign: src, nisar_update: $("#nisar_update").checked, nisar_assumed: +$("#s_nisar_assumed").value, model: $("#model").value, _token: ld.token };
    const res = await fetch(src === "sim" ? "/api/season" : "/api/replay", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    if (!res.ok) throw new Error(await res.text());
    ld.stopPolling();
    next = await loaderReceive(ld, res);
  } catch (e) {
    ld.stopPolling();
    if (seqNo !== runSeq) return;
    loaderFail(ld, e.message);
    $("#status").textContent = `Failed: ${e.message}`;
    $("#run").disabled = $("#reseed").disabled = false;
    return;
  }
  if (seqNo !== runSeq) return;
  R = next;
  loaderStep(ld, ld.steps.length - 1, 0.97);
  await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));   // let the card repaint
  if (view === "3d" && !village && webglOK()) { try { await import("./scene3d.js"); } catch { /* show3D reports it */ } }
  PH.cache = new Map();
  passIdx = new Map();
  for (const p of R.passes) { const k = p.f * R.T + p.d; if (!passIdx.has(k)) passIdx.set(k, []); passIdx.get(k).push(p); }
  asks = new Map();
  if (sel !== null && sel >= R.n) sel = null;
  qcDrawn = null;
  CF.data = null; CF.drawn = null; MH.data = null; MH.drawn = null;
  if (replay() && (view === "advice" || view === "experiment")) {
    view = "3d";
    document.querySelectorAll(".seg button").forEach((x) => x.classList.toggle("on", x.dataset.view === "3d"));
  }
  if (!replay() && view === "qc") {
    view = "3d";
    document.querySelectorAll(".seg button").forEach((x) => x.classList.toggle("on", x.dataset.view === "3d"));
  }
  $("#day").max = R.T - 1;
  day = Math.min(day, R.T - 1);
  const lay = R.layout;
  document.body.classList.toggle("replay", replay());
  $("#badge").textContent = replay() ? (R.campaign.synthetic ? "Synthetic campaign data" : "Campaign data") : "Simulated data";
  $("#status").textContent = (replay() ? `Replay: ${R.campaign.label} · ` : "") + `${lay.label}${lay.surveyed ? " (surveyed)" : ""} · ${R.n} fields, ${R.loggers.length} loggers, ${R.T} days, ` +
    `${R.passes.length} satellite field passes (${SENSOR_ORDER.map((k) => `${SENSORS[k].label} ${R.passes.filter((p) => p.s === k).length}`).join(", ")}) · computed in ${R.seconds} s`;
  $("#run").disabled = $("#reseed").disabled = false;
  seasonSummary();
  if (village) village.setData(R);
  setDay(day);
  if (sel !== null) select(sel);
  if (seqNo === runSeq) loaderEnd(ld);
}

function togglePlay() {
  if (timer) return stop();
  if (day >= R.T - 1) setDay(0);
  $("#play").textContent = "❚❚";
  timer = setInterval(() => { if (day >= R.T - 1) return stop(); setDay(day + 1); }, +$("#speed").value);
}
function stop() { clearInterval(timer); timer = null; $("#play").textContent = "▶"; }

function setDay(d) {
  day = d;
  if (sel !== null) setTimeout(cropLine, 0);
  if (PH.tab === "phone" && sel !== null) setTimeout(phoneShow, 0);
  $("#day").value = d;
  $("#date").textContent = `${fmtDate(d)} · day ${d + 1}/${R.T}`;
  $("#askday").textContent = fmtDate(d);
  kpis(); draw(); drawChart();
}
function fmtDate(d) {
  const t = new Date(R.start_date + "T00:00:00Z");
  t.setUTCDate(t.getUTCDate() + d);
  return t.toLocaleDateString("en-GB", { day: "numeric", month: "short", timeZone: "UTC" });
}

// ── colours ────────────────────────────────────────────────────────────────
function readColors() {
  const cs = getComputedStyle(document.documentElement);
  for (const k of ["ink", "ink2", "muted", "line", "surface", "accent", "accent-soft", "warn", "alert", "wrong", "dry", "mid", "wet",
    "cat1", "cat2", "cat3", "cat4", "cat5", "cat6", "cat7", "cat8", "seq1", "seq2", "seq3"])
    colors[k] = cs.getPropertyValue(`--${k}`).trim();
}
function hex(c) { const n = parseInt(c.slice(1), 16); return [n >> 16, (n >> 8) & 255, n & 255]; }
function mix(a, b, t) {
  const A = hex(a), B = hex(b);
  return `rgb(${A.map((x, i) => Math.round(x + (B[i] - x) * t)).join(",")})`;
}
function depthColor(h) {
  if (h === null || h === undefined || Number.isNaN(h)) return colors.line;
  if (h > 0) return mix(colors.mid, colors.wet, Math.min(1, Math.sqrt(h / 80)));
  return mix(colors.mid, colors.dry, Math.min(1, Math.sqrt(-h / 200)));
}
function leakColor(sp) {
  const t = Math.min(1, Math.max(0, Math.log(sp / 1.5) / Math.log(25 / 1.5)));
  return mix("#efe9f8", "#4a3aa7", t);
}

// ── map: field polygons in local metres (R.layout), drawn to fit the canvas ────
const fieldM3 = (f, mm) => (mm * R.fields[f].area_m2) / 1000;   // mm of water over the field -> m³
function computeLayout() {
  const W = $("#map").clientWidth, pad = 10, top = 22, L = R.layout;
  let x0 = Infinity, x1 = -Infinity, z0 = Infinity, z1 = -Infinity;
  const ext = (x, z) => { x0 = Math.min(x0, x); x1 = Math.max(x1, x); z0 = Math.min(z0, z); z1 = Math.max(z1, z); };
  L.fields.forEach((fl) => fl.poly.forEach(([x, z]) => ext(x, z)));
  L.tubewells.forEach((t) => ext(t.x - 4, t.z));
  const s = (W - 2 * pad) / (x1 - x0);
  const P = (x, z) => [pad + (x - x0) * s, top + (z - z0) * s];
  const polys = L.fields.map((fl) => fl.poly.map(([x, z]) => P(x, z)));
  const bbox = polys.map((pp) => ({ x0: Math.min(...pp.map((p) => p[0])), x1: Math.max(...pp.map((p) => p[0])),
                                     y0: Math.min(...pp.map((p) => p[1])), y1: Math.max(...pp.map((p) => p[1])) }));
  return { W, H: top + (z1 - z0) * s + pad, s, P, polys, bbox,
           cents: L.fields.map((fl) => P(fl.c[0], fl.c[1])),
           size: R.fields.map((F) => Math.sqrt(F.area_m2) * s),
           loggers: (R.loggers || []).map((lg) => P(lg.x, lg.z)) };
}
function tracePoly(g, pp) { g.beginPath(); g.moveTo(pp[0][0], pp[0][1]); for (let i = 1; i < pp.length; i++) g.lineTo(pp[i][0], pp[i][1]); g.closePath(); }
function traceLine(g, pts) { g.beginPath(); pts.forEach(([x, z], i) => { const [cx, cy] = layout.P(x, z); i ? g.lineTo(cx, cy) : g.moveTo(cx, cy); }); }

function draw() {
  if (!R) return;
  $("#qcpanel").hidden = view !== "qc";
  $("#planpanel").hidden = view !== "plan";
  $("#cfpanel").hidden = view !== "advice";
  $("#mhpanel").hidden = view !== "methane";
  $("#expanel").hidden = view !== "experiment";
  if (view === "experiment") {
    $("#map").hidden = true; $("#tip").hidden = true;
    if (village) village.setVisible(false);
    exPanel();
    return;
  }
  if (view === "methane") mhPanel();
  if (view === "advice") cfPanel();
  if (view === "plan" && !planReady()) return;
  if (view === "qc") {
    $("#map").hidden = true; $("#tip").hidden = true;
    if (village) village.setVisible(false);
    renderQC();
    return;
  }
  if (view === "3d") { show3D(); return; }
  $("#map").hidden = false;
  if (village) village.setVisible(false);
  layout = computeLayout();
  const cv = $("#map"), dpr = window.devicePixelRatio || 1, L = layout;
  cv.width = L.W * dpr; cv.height = L.H * dpr; cv.style.height = `${L.H}px`;
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, L.W, L.H);
  const tws = R.layout.tubewells, running = tws.map(() => false), toDate = tws.map(() => 0);
  for (let f = 0; f < R.n; f++) {
    const t = R.fields[f].tw;
    if (R.irr[f][day] > 0) running[t] = true;
    for (let d = 0; d <= day; d++) toDate[t] += fieldM3(f, R.irr[f][d]);
  }
  g.lineCap = "round"; g.lineJoin = "round";
  for (const r of R.layout.roads) { traceLine(g, r.pts); g.strokeStyle = colors.line; g.lineWidth = Math.max(2, r.width * L.s); g.stroke(); }
  for (const ch of R.layout.channels) {
    traceLine(g, ch.pts); g.strokeStyle = running[ch.tw] ? colors.accent : colors["accent-soft"];
    g.lineWidth = Math.max(2, ch.width * L.s * 0.6); g.stroke();
  }
  for (let f = 0; f < R.n; f++) (view === "plan" ? drawPlanCell(g, f) : view === "advice" ? drawCfCell(g, f) : view === "methane" ? drawMhCell(g, f) : drawCell(g, f));
  // tubewells: numbered pump markers; pumping today in the accent colour
  g.textAlign = "center"; g.textBaseline = "middle";
  tws.forEach((t, i) => {
    const [x, y] = L.P(t.x, t.z);
    g.fillStyle = running[i] ? colors.accent : colors.ink2; g.beginPath(); g.roundRect(x - 8, y - 8, 16, 16, 3); g.fill();
    g.fillStyle = "#fff"; g.font = "700 10px system-ui"; g.fillText(String(i + 1), x, y + 0.5);
  });
  // header: water pumped so far per tubewell
  g.textAlign = "left"; g.textBaseline = "top"; g.font = "11px system-ui"; g.fillStyle = colors.ink2;
  g.fillText(tws.map((t, i) => `${i + 1}: ${Math.round(toDate[i]).toLocaleString()} m³`).join("   ·   ") + "   pumped so far", 10, 4, L.W - 20);
  if (view === "plan") { drawPlanPipes(g); return; }
  if (view === "advice") return;
  if (view === "methane") { (R.loggers || []).forEach((lg, i) => { const [x, y] = L.loggers[i]; g.beginPath(); g.arc(x, y, 3.5, 0, 7); g.fillStyle = "#fff"; g.fill(); g.lineWidth = 1.2; g.strokeStyle = colors.ink; g.stroke(); }); return; }
  // loggers
  (R.loggers || []).forEach((lg, i) => {
    const [x, y] = L.loggers[i], on = qcOnDay(lg.id, day);
    const dead = on.some((q) => q.severity === "high"), warn = on.some((q) => q.severity === "medium");
    g.beginPath(); g.arc(x, y, 4, 0, 7); g.fillStyle = dead ? colors.wrong : warn ? colors.warn : "#fff"; g.fill();
    g.lineWidth = 1.5; g.strokeStyle = colors.ink; g.stroke();
  });
}

function drawCell(g, f) {
  const L = layout, pp = L.polys[f], bb = L.bbox[f], [cx, cy] = L.cents[f], sz = L.size[f];
  const t = R.truth[f][day], b = R.belief.h[f][day], pw = R.belief.p_wet[f][day];
  g.save(); tracePoly(g, pp); g.clip();
  const w = bb.x1 - bb.x0, h = bb.y1 - bb.y0;
  if (!planted(f, day)) {
    g.fillStyle = colors.line; g.fillRect(bb.x0, bb.y0, w, h);
  } else if (view === "truth") {
    g.fillStyle = depthColor(t); g.fillRect(bb.x0, bb.y0, w, h);
    if (t === null) stripes(g, bb.x0, bb.y0, w, h);   // no logger in this field
  } else if (view === "belief") {
    g.fillStyle = depthColor(b); g.fillRect(bb.x0, bb.y0, w, h);
    if (pw > 0.25 && pw < 0.75) stripes(g, bb.x0, bb.y0, w, h);
  } else {  // X-ray and leak rate: truth left of the centroid, model right
    const [l, r] = view === "leak" ? [R.fields[f].sp_true === null ? colors.line : leakColor(R.fields[f].sp_true), leakColor(R.fields[f].sp_model)] : [depthColor(t), depthColor(b)];
    g.fillStyle = l; g.fillRect(bb.x0, bb.y0, cx - bb.x0, h);
    if ((view === "leak" ? R.fields[f].sp_true : t) === null) stripes(g, bb.x0, bb.y0, cx - bb.x0, h);   // no logger
    g.fillStyle = r; g.fillRect(cx, bb.y0, bb.x1 - cx, h);
    if (view !== "leak" && pw > 0.25 && pw < 0.75) stripes(g, cx, bb.y0, bb.x1 - cx, h);
    g.fillStyle = colors.surface; g.fillRect(cx - 0.5, bb.y0, 1, h);
  }
  g.restore();
  tracePoly(g, pp); g.strokeStyle = colors.surface; g.lineWidth = 1.2; g.setLineDash([]); g.stroke();
  if (planted(f, day) && view !== "leak") overlays(g, f, pp, cx, cy, sz);
  if (f === sel) { tracePoly(g, pp); g.strokeStyle = colors.accent; g.lineWidth = 3; g.stroke(); }
}
function stripes(g, x, y, w, h) {
  g.save(); g.beginPath(); g.rect(x, y, w, h); g.clip();
  g.strokeStyle = "rgba(255,255,255,.55)"; g.lineWidth = 2;
  for (let k = -h; k < w; k += 6) { g.beginPath(); g.moveTo(x + k, y + h); g.lineTo(x + k + h, y); g.stroke(); }
  g.restore();
}
function overlays(g, f, pp, cx, cy, sz) {
  const r = Math.min(9, Math.max(3, sz * 0.1)), o = sz * 0.25;
  const ps = passesAt(f, day);
  if (ps.length) {   // outline in the sensor's colour (dashed red if any pass that day was wrong)
    const wrong = ps.some((q) => isWrong(q));
    tracePoly(g, pp); g.lineWidth = 2;
    if (!wrong) { g.strokeStyle = SENSORS[ps[0].s].color; g.setLineDash([]); } else { g.strokeStyle = colors.wrong; g.setLineDash([4, 3]); }
    g.stroke(); g.setLineDash([]);
    if (ps.some((q) => q.flag)) {
      g.fillStyle = colors.warn; g.beginPath(); g.arc(cx + o, cy - o, r * 1.2, 0, 7); g.fill();
      g.fillStyle = "#fff"; g.font = `700 ${r * 1.8}px system-ui`; g.textAlign = "center"; g.textBaseline = "middle";
      g.fillText("!", cx + o, cy - o + 0.5);
    }
  }
  const dp = R.dry_pred[f][day], dt = R.dry_true[f][day];
  if (dp !== null && dp <= R.alert_days) {
    g.beginPath(); g.arc(cx - o, cy + o, r, 0, 7);
    if (dt !== null && dt <= R.alert_days) { g.fillStyle = colors.alert; g.fill(); } else { g.strokeStyle = colors.alert; g.lineWidth = 2; g.stroke(); }
  }
  if (R.irr[f][day] > 0) {
    const x = cx - o, y = cy - o;
    g.beginPath(); g.moveTo(x, y - r); g.lineTo(x + r, y + r * 0.8); g.lineTo(x - r, y + r * 0.8); g.closePath();
    if (R.logged[f][day]) { g.fillStyle = colors.accent; g.fill(); } else { g.strokeStyle = colors.accent; g.lineWidth = 1.5; g.stroke(); }
  }
}

function inPoly(x, y, pp) {
  let c = false;
  for (let i = 0, j = pp.length - 1; i < pp.length; j = i++)
    if ((pp[i][1] > y) !== (pp[j][1] > y) && x < pp[j][0] + ((y - pp[j][1]) * (pp[i][0] - pp[j][0])) / (pp[i][1] - pp[j][1])) c = !c;
  return c;
}
function mapXY(e) { const rect = $("#map").getBoundingClientRect(); return [e.clientX - rect.left, e.clientY - rect.top]; }
function hitField(e) {
  if (!R || !layout) return null;
  const [mx, my] = mapXY(e);
  for (let f = 0; f < R.n; f++) if (inPoly(mx, my, layout.polys[f])) return f;
  return null;
}
function hitLogger(e) {
  if (!R || !layout) return null;
  const [mx, my] = mapXY(e);
  const i = layout.loggers.findIndex(([x, y]) => Math.hypot(mx - x, my - y) < 7);
  return i < 0 ? null : i;
}
function hoverMap(e) {
  if (view === "plan") return planTip(e);
  if (view === "advice") return cfTip(e);
  if (view === "methane") return mhTip(e);
  const tip = $("#tip"), li = hitLogger(e), f = li !== null ? R.loggers[li].field : hitField(e);
  if (f === null) { tip.hidden = true; return; }
  const F = R.fields[f], t = R.truth[f][day], pw = R.belief.p_wet[f][day];
  const ps = passesAt(f, day);
  const qcl = li !== null ? qcOnDay(R.loggers[li].id, day) : [];
  tip.innerHTML = (li !== null ? `<b>Logger ${R.loggers[li].id}</b> · reading ${fmtH(R.loggers[li].reading[day])}<br>` +
    qcl.map((q) => `<span style="color:var(--${q.severity === "high" ? "wrong" : "warn"})">QC: ${q.title}</span><br>`).join("") : "") +
    `<b>Field ${F.id}</b> · ${F.ftype}${F.logs ? "" : " · does not log"} · ${(F.area_m2 / 1e4).toFixed(2)} ha<br>` +
    (!planted(f, day) ? "not planted yet" :
      `${t === null ? (replay() ? "No logger in this field" : "–") : `${replay() ? "Logger" : "True water"}: <b>${fmtH(t)}</b>`}<br>Model: P(ponded) ${(pw * 100).toFixed(0)} %, ${fmtH(R.belief.h[f][day])}`) +
    `<br>S&P: ${replay() ? "from logger" : "true"} ${F.sp_true ?? "–"} · model ${F.sp_model} mm/d` +
    ps.map((p) => `<br>${SENSORS[p.s].label} said <b>${p.wet ? "wet" : "dry"}</b>${isWrong(p) ? " (wrong)" : p.truth === null ? " (no logger to check)" : ""}${p.flag ? " · model doubts it" : ""}`).join("") +
    nisarNote(f, day);
  const rect = $("#map").getBoundingClientRect();
  tip.hidden = false;
  const x = e.clientX - rect.left + 14, y = e.clientY - rect.top + 14;
  tip.style.left = `${Math.min(x, rect.width - tip.offsetWidth - 4)}px`;
  tip.style.top = `${y}px`;
}
function fmtH(h) { return h === null || h === undefined ? "–" : h > 0 ? `${h.toFixed(0)} mm ponded` : `${(-h).toFixed(0)} mm below surface`; }

// ── 3D village ─────────────────────────────────────────────────────────────
function webglOK() {
  try { const c = document.createElement("canvas"); return !!(c.getContext("webgl2") || c.getContext("webgl")); } catch { return false; }
}
function show3D() {
  $("#map").hidden = true;
  $("#qcpanel").hidden = true;
  $("#tip").hidden = true;
  if (village) { village.setVisible(true); village.setDay(day); village.setSelected(sel); return; }
  if (loading3D) return;
  loading3D = true;
  import("./scene3d.js").then((m) => {
    if (!webglOK()) throw new Error("WebGL is not available in this browser");
    village = new m.Village3D($("#scene3d"), { onSelect: (f) => select(f) });
    window.paddyTwin3D = village;   // scripting hook (camera tours, screenshots, future sensors)
    if (R) village.setData(R);
    if (view === "3d") show3D();
  }).catch((e) => {
    $("#status").textContent = `3D view unavailable (${e.message}); showing the 2D map.`;
    view = "xray";
    document.querySelectorAll(".seg button").forEach((x) => x.classList.toggle("on", x.dataset.view === "xray"));
    draw();
  });
}

// ── KPIs and legend ────────────────────────────────────────────────────────
function kpis() {
  let inS = 0, wetT = 0, wetB = 0, agree = 0, alerts = 0, hits = 0, m3 = 0, m3d = 0;
  for (let f = 0; f < R.n; f++) {
    for (let d = 0; d <= day; d++) m3 += fieldM3(f, R.irr[f][d]);
    m3d += fieldM3(f, R.irr[f][day]);
    const t = R.truth[f][day];
    if (t === null) continue;
    inS++;
    const bw = R.belief.p_wet[f][day] > 0.5;
    wetT += t > 0; wetB += bw; agree += (t > 0) === bw;
    const dp = R.dry_pred[f][day], dt = R.dry_true[f][day];
    if (dp !== null && dp <= R.alert_days) { alerts++; hits += dt !== null && dt <= R.alert_days; }
  }
  const ps = R.passes.filter((p) => p.d === day);
  const wrong = ps.filter((p) => isWrong(p)).length, flags = ps.filter((p) => p.flag).length;
  const today = SENSOR_ORDER.filter((k) => ps.some((p) => p.s === k)).map((k) => SENSORS[k].label);
  const items = [
    [replay() ? "Logger fields ponded" : "Fields ponded", `${wetT}`, `model believes ${wetB} (of ${inS} ${replay() ? "with a logger" : "planted"})`],
    ["Model right today", inS ? `${Math.round((100 * agree) / inS)} %` : "–", "ponded or not, per field"],
    ["Irrigate-by-tomorrow alerts", `${alerts}`, alerts ? `${hits} correct` : "none today"],
    ["Satellite today", ps.length ? today.join(" + ") : "no pass", ps.length ? `${ps.length} field looks · ${wrong} wrong · ${flags} doubted` : "S2 ~5 d (if clear) · S1 6 d · NISAR 12 d"],
    ["Water pumped", `${Math.round(m3).toLocaleString()} m³`, `${Math.round(m3d).toLocaleString()} m³ today`],
  ];
  $("#kpis").innerHTML = items.map(([l, n, s]) => `<div class="kpi"><div class="l">${l}</div><div class="n">${n}</div><div class="s">${s}</div></div>`).join("");
  legend();
}
function legend() {
  const el = $("#legend");
  if (view === "3d") {
    el.innerHTML = `<span>Water: blue when deep, olive when shallow · soil darkens when wet and lightens and cracks as it dries ·
      rice grows with the simulated LAI and ripens late in the season · a channel fills and its tubewell spouts on days a field
      in that area is irrigated · <span class="k k-pass"></span> Sentinel-2 pass (red if wrong) · <b style="color:var(--warn)">!</b> pass the model doubts ·
      <span class="k k-alert"></span> irrigate-by-tomorrow alert</span>`;
    return;
  }
  if (view === "plan") { el.innerHTML = planLegend(); return; }
  if (view === "advice") { el.innerHTML = cfLegend(); return; }
  if (view === "methane") { el.innerHTML = mhLegend(); return; }
  if (view === "experiment") { el.innerHTML = ""; return; }
  if (view === "qc") {
    el.innerHTML = `<span>${Object.entries(SEV).map(([k, l]) => `<span class="sev sev-${k}">${l}</span>`).join(" ")}
      · loggers are checked against their own record and against the model (which never sees them) · click an issue or a logger
      to open its field · thresholds are in app/qc.py</span>`;
    return;
  }
  if (view === "leak") {
    el.innerHTML = `<span>Leak rate (effective S&P), left half true, right half model:</span>
      <span><div class="bar" style="background:linear-gradient(90deg,${leakColor(1.5)},${leakColor(5)},${leakColor(25)})"></div>
      <div class="ticks"><span>1.5</span><span>5</span><span>25 mm/d</span></div></span>`;
    return;
  }
  const stops = [-200, -100, -30, 0, 10, 30, 80].map((h) => depthColor(h)).join(",");
  el.innerHTML = `<span>${view === "xray" ? "Left half truth, right half model belief" : view === "truth" ? "True water level" : "Model belief"}:</span>
    <span><div class="bar" style="background:linear-gradient(90deg,${stops})"></div>
    <div class="ticks"><span>200 mm below</span><span>surface</span><span>80 mm ponded</span></div></span>
    <span><span class="k k-pass"></span> pass · <span class="k k-wrong"></span> wrong pass · <b style="color:var(--warn)">!</b> doubted ·
    <span class="k k-alert"></span> alert · <span class="k k-irr"></span> irrigated</span>`;
}

// ── field detail ───────────────────────────────────────────────────────────
function select(f) {
  sel = f;
  const F = R.fields[f];
  const lgs = (R.loggers || []).filter((lg) => lg.field === f);
  $("#ftitle").textContent = `Field ${F.id} · ${R.layout.tubewells[F.tw].name}`;
  const nIrr = R.irr[f].filter((x) => x > 0).length, nLog = R.logged[f].filter((x) => x > 0).length;
  $("#finfo").innerHTML = `<div class="grid">
    <span>Farmer</span><span>${F.ftype}, ${F.logs ? `logs in the app (${nLog} of ${nIrr} irrigations logged)` : `<b>does not log</b> (${nIrr} irrigations)`}</span>
    <span>Planted</span><span>${fmtDate(Math.max(0, Math.floor(F.tp)))}</span>
    <span>Leak (S&P)</span><span>${replay() ? (F.sp_true === null ? "no logger estimate" : `from its logger ${F.sp_true} mm/d`) : `true ${F.sp_true} mm/d`} · model ${F.sp_model} mm/d</span>
    <span>Area</span><span>${(F.area_m2 / 1e4).toFixed(2)} ha</span>
    <span>Water</span><span>${Math.round(F.irr_total)} mm this season (${Math.round(fieldM3(f, F.irr_total))} m³)</span>
    <span>Crop</span><span id="fcrop"></span>
    <span>Satellites</span><span id="fsens"></span>
    ${lgs.map((lg) => `<span>Logger</span><span>${lg.id}: today ${fmtH(lg.reading[day])} (grey dots on the chart)</span>`).join("")}
    ${replay() && R.qc ? R.qc.issues.filter((q) => q.field === f).map((q) => `<span class="sev sev-${q.severity}">${SEV[q.severity]}</span><span>${q.title} (${q.start_date} to ${q.end_date})</span>`).join("") : ""}</div>`;
  $("#ask").hidden = replay() || PH.tab === "phone";
  $("#answer").innerHTML = "";
  cropLine();
  $("#dtabs").hidden = false;
  phoneShow();
  draw(); drawChart();
}

function drawChart() {
  const svg = $("#chart");
  if (!R || sel === null) { svg.innerHTML = ""; return; }
  const f = sel, W = svg.clientWidth || 360, H = 272, m = { l: 34, r: 8, t: 46, b: 34 };
  const ymin = -250, ymax = 150;
  const X = (d) => m.l + ((W - m.l - m.r) * d) / (R.T - 1);
  const Y = (h) => m.t + ((H - m.t - m.b) * (ymax - Math.max(ymin, Math.min(ymax, h)))) / (ymax - ymin);
  const seg = (arr) => {
    let p = "", on = false;
    arr.forEach((v, d) => { if (v === null) { on = false; return; } p += `${on ? "L" : "M"}${X(d).toFixed(1)},${Y(v).toFixed(1)}`; on = true; });
    return p;
  };
  const lo = R.belief.q05[f], hi = R.belief.q95[f];
  let band = "", start = null;
  const flush = (end) => {
    if (start === null) return;
    let p = `M${X(start)},${Y(hi[start])}`;
    for (let d = start + 1; d <= end; d++) p += `L${X(d)},${Y(hi[d])}`;
    for (let d = end; d >= start; d--) p += `L${X(d)},${Y(lo[d])}`;
    band += p + "Z"; start = null;
  };
  for (let d = 0; d < R.T; d++) { if (lo[d] !== null && start === null) start = d; if (lo[d] === null) flush(d - 1); }
  flush(R.T - 1);
  let g = "";
  for (const h of [-200, -100, 0, 100]) {
    g += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(h)}" y2="${Y(h)}" stroke="${h === 0 ? colors.ink2 : colors.line}" stroke-width="${h === 0 ? 1 : 0.8}"/>`;
    g += `<text x="${m.l - 4}" y="${Y(h) + 3}" text-anchor="end">${h}</text>`;
  }
  for (let d = 0; d < R.T; d += 14) g += `<text x="${X(d)}" y="${H - m.b + 12}" text-anchor="middle">${fmtDate(d)}</text>`;
  g += `<text x="${m.l}" y="10">water level, mm (0 = soil surface)</text>`;
  g += `<path d="${band}" fill="${colors["accent-soft"]}" opacity=".9"/>`;
  g += `<path d="${seg(R.belief.q50[f])}" fill="none" stroke="${colors.accent}" stroke-width="2"/>`;
  g += `<path d="${seg(R.truth[f])}" fill="none" stroke="${colors.ink}" stroke-width="1.6"/>`;
  for (const lg of (R.loggers || []).filter((l) => l.field === f))
    lg.reading.forEach((v, d) => { if (v !== null) g += `<circle cx="${X(d)}" cy="${Y(v)}" r="1.8" fill="${colors.muted}"><title>${fmtDate(d)}: logger ${lg.id} reads ${v} mm</title></circle>`; });
  // irrigations (bottom strip) and passes (top strip)
  for (let d = 0; d < R.T; d++) {
    const mm = R.irr[f][d];
    if (mm > 0) {
      const h = Math.min(16, mm / 6), x = X(d) - 2.5;
      g += `<rect x="${x}" y="${H - m.b - h}" width="5" height="${h}" rx="1" ${R.logged[f][d] ? `fill="${colors.accent}"` : `fill="none" stroke="${colors.accent}"`}><title>${fmtDate(d)}: irrigated ${mm.toFixed(0)} mm${R.logged[f][d] ? " (logged)" : " (not logged)"}</title></rect>`;
    }
    for (const p of passesAt(f, d)) {   // one row per sensor; filled = said wet, hollow = said dry, red = wrong
      const c = !isWrong(p) ? SENSORS[p.s].color : colors.wrong, x = X(d), y = m.t - 4 - 8 * (2 - SENSOR_ORDER.indexOf(p.s));
      g += `<path d="M${x - 3.5},${y - 5}L${x + 3.5},${y - 5}L${x},${y + 1}Z" ${p.wet ? `fill="${c}"` : `fill="none" stroke="${c}"`}><title>${fmtDate(d)}: ${SENSORS[p.s].label} said ${p.wet ? "wet" : "dry"}${!isWrong(p) ? "" : " (wrong)"}; model prior P(ponded) ${(p.prior * 100).toFixed(0)} %${p.flag ? " — doubted" : ""}</title></path>`;
    }
  }
  for (let d = 0; d < R.T; d++) {
    const p = nisarNext(f, d);
    if (p && R.belief.q50[f][d] !== null)
      g += `<circle cx="${X(d)}" cy="${Y(R.belief.q50[f][d])}" r="3.2" fill="${SENSORS.nisar.color}" stroke="${colors.surface}" stroke-width="1"><title>${fmtDate(d)}: advice updated with the NISAR pass of ${fmtDate(d + 1)}, 06:00 (said ${p.wet ? "wet" : "dry"})</title></circle>`;
  }
  for (const a of asks.get(f) || []) {
    const x = X(a.day);
    g += `<line x1="${x - 3}" x2="${x - 3}" y1="${Y(a.before.h[0])}" y2="${Y(a.before.h[2])}" stroke="${colors.accent}" stroke-width="3" opacity=".5"/>`;
    g += `<line x1="${x + 3}" x2="${x + 3}" y1="${Y(a.after.h[0])}" y2="${Y(a.after.h[2])}" stroke="${colors.warn}" stroke-width="3"/>`;
  }
  g += `<line x1="${X(day)}" x2="${X(day)}" y1="${m.t}" y2="${H - m.b}" stroke="${colors.ink2}" stroke-dasharray="3 3"/>`;
  g += `<g id="hover"></g>`;
  g += `<g font-size="10" transform="translate(${m.l},${H - 8})">
    <line x1="0" x2="14" y1="-3" y2="-3" stroke="${colors.ink}" stroke-width="1.6"/><text x="18" y="0">${replay() ? "logger" : "truth"}</text>
    <line x1="56" x2="70" y1="-3" y2="-3" stroke="${colors.accent}" stroke-width="2"/><text x="74" y="0">model (90 % range shaded)</text>
    </g><g font-size="8">${SENSOR_ORDER.map((k, i) => `<text x="${m.l - 4}" y="${m.t - 4 - 8 * (2 - i) + 0}" text-anchor="end" style="fill:${SENSORS[k].color}">${k === "nisar" ? "NISAR" : k.toUpperCase()}</text>`).join("")}</g>`;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.innerHTML = g;
  svg.onmousemove = (e) => {
    const r = svg.getBoundingClientRect(), d = Math.round(((e.clientX - r.left - m.l) / (W - m.l - m.r)) * (R.T - 1));
    if (d < 0 || d >= R.T) return;
    svg.querySelector("#hover").innerHTML = `<line x1="${X(d)}" x2="${X(d)}" y1="${m.t}" y2="${H - m.b}" stroke="${colors.muted}"/>`;
    const tip = $("#ctip"), t = R.truth[f][d];
    tip.innerHTML = `<b>${fmtDate(d)}</b><br>` + (!planted(f, d) ? "not planted" :
      `Truth: ${fmtH(t)}<br>Model: ${fmtH(R.belief.q50[f][d])} (90 %: ${R.belief.q05[f][d]} to ${R.belief.q95[f][d]})<br>P(ponded) ${(R.belief.p_wet[f][d] * 100).toFixed(0)} %` +
      (R.dry_pred[f][d] !== null ? `<br>Dry in: model ${R.dry_pred[f][d].toFixed(1)} d · truth ${R.dry_true[f][d] === null ? "–" : R.dry_true[f][d].toFixed(1)} d` : ""));
    tip.hidden = false;
    const pr = $("#detail").getBoundingClientRect();
    tip.style.left = `${Math.min(e.clientX - pr.left + 12, pr.width - tip.offsetWidth - 6)}px`;
    tip.style.top = `${e.clientY - pr.top + 12}px`;
  };
  svg.onmouseleave = () => { $("#ctip").hidden = true; const h = svg.querySelector("#hover"); if (h) h.innerHTML = ""; };
  svg.onclick = (e) => {
    const r = svg.getBoundingClientRect(), d = Math.round(((e.clientX - r.left - m.l) / (W - m.l - m.r)) * (R.T - 1));
    if (d >= 0 && d < R.T) { stop(); setDay(d); }
  };
}

async function ask(kind) {
  if (sel === null) return;
  const out = $("#answer");
  out.textContent = "Asking…";
  const res = await fetch("/api/ask", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ run: R.run, field: sel, day, kind }) });
  const a = await res.json();
  if (!res.ok) { out.textContent = a.detail || "Could not ask."; return; }
  asks.set(sel, [...(asks.get(sel) || []).filter((x) => x.day !== a.day), a]);
  const row = (l, b, af, t) => `<tr><td>${l}</td><td>${b}</td><td><b>${af}</b></td><td>${t}</td></tr>`;
  const rng = (q) => `${q[1].toFixed(0)} (${q[0].toFixed(0)} to ${q[2].toFixed(0)})`;
  const dry = (q) => `${q[1].toFixed(1)} (${q[0].toFixed(1)}–${q[2].toFixed(1)})`;
  out.innerHTML = `Farmer says: <b>${a.answer}</b>
    <table><tr><th></th><th>before</th><th>after</th><th>truth</th></tr>
    ${row("P(ponded)", `${(a.before.p_wet * 100).toFixed(0)} %`, `${(a.after.p_wet * 100).toFixed(0)} %`, a.truth_h > 0 ? "ponded" : "not ponded")}
    ${row("Water level, mm (90 %)", rng(a.before.h), rng(a.after.h), a.truth_h.toFixed(0))}
    ${row("Dry in, days (90 %)", dry(a.before.dry_in), dry(a.after.dry_in), a.truth_dry_in.toFixed(1))}</table>
    ${a.nisar ? `<span style="color:${SENSORS.nisar.color}">"Before" already includes the next morning's NISAR pass (said ${a.nisar}).</span><br>` : ""}
    <span class="muted">Farmers answer wrongly 5 % of the time (Experiment 13). Bars on the chart: before (blue) and after (orange).</span>`;
  drawChart();
}

// ── season summary ─────────────────────────────────────────────────────────
function seasonSummary() {
  let agree = 0, n = 0, alerts = 0, hits = 0, needed = 0, caught = 0;
  const errs = [];
  for (let f = 0; f < R.n; f++) for (let d = 0; d < R.T; d++) {
    const t = R.truth[f][d];
    if (t === null) continue;
    n++; agree += (t > 0) === (R.belief.p_wet[f][d] > 0.5);
    const dp = R.dry_pred[f][d], dt = R.dry_true[f][d];
    const al = dp !== null && dp <= R.alert_days, dryBy = dt !== null && dt <= R.alert_days, need = t > 0 && dryBy;
    alerts += al; hits += al && dryBy; needed += need; caught += al && need;
    if (dp !== null && dt !== null && t > 0) errs.push(Math.abs(dp - dt));
  }
  errs.sort((a, b) => a - b);
  const bySensor = (k) => R.passes.filter((p) => p.s === k);
  const spErr = R.fields.filter((F) => F.sp_true).map((F) => Math.abs(Math.log(F.sp_model / F.sp_true))).sort((a, b) => a - b);
  const water = R.irr.reduce((s, row, f) => s + fieldM3(f, row.reduce((a, b) => a + b, 0)), 0);
  const rows = [
    ["Model right about ponded / not ponded", `${((100 * agree) / n).toFixed(0)} % of field-days`],
    ["Alerts right (field dry by tomorrow)", alerts ? `${((100 * hits) / alerts).toFixed(0)} % of ${alerts}` : "–"],
    ["Ponded fields drying by tomorrow that got an alert", needed ? `${((100 * caught) / needed).toFixed(0)} % of ${needed}` : "–"],
    ["Next-dry forecast error (median)", errs.length ? `${errs[Math.floor(errs.length / 2)].toFixed(1)} days` : "–"],
    ...SENSOR_ORDER.map((k) => {
      const P = bySensor(k).filter((p) => p.truth !== null && p.truth !== undefined), fl = P.filter((p) => p.flag), w = P.filter((p) => isWrong(p)).length;
      return [`${SENSORS[k].label}: wrong · doubted-and-wrong${replay() ? ` (${P.length} passes over loggers)` : ""}`, P.length ? `${((100 * w) / P.length).toFixed(0)} % · ${fl.length ? ((100 * fl.filter((p) => isWrong(p)).length) / fl.length).toFixed(0) + " % of " + fl.length : "–"}` : "–"];
    }),
    ...(R.nisar_update.enabled && R.nisar_update.ponded_bacc_before != null ? [["NISAR answer update: ponded now · next-dry error",
      `${(R.nisar_update.ponded_bacc_before * 100).toFixed(0)} → ${(R.nisar_update.ponded_bacc_after * 100).toFixed(0)} % · ` +
      (R.nisar_update.dry_err_before != null ? `${R.nisar_update.dry_err_before.toFixed(2)} → ${R.nisar_update.dry_err_after.toFixed(2)} d` : "–") +
      ` (${R.nisar_update.n_scored ?? R.nisar_update.n} field-mornings${replay() ? " with a logger" : ""})`]] : []),
    [replay() ? "Leak rate: model vs logger estimate (median)" : "Leak-rate error (median)", spErr.length ? `${((Math.exp(spErr[Math.floor(spErr.length / 2)]) - 1) * 100).toFixed(0)} %${replay() ? ` (${spErr.length} logger fields)` : ""}` : "–"],
    ["Water pumped this season", `${Math.round(water).toLocaleString()} m³`],
  ];
  if (replay()) {   // the loggers are the only truth: drop rows that the logger-scored rows below replace
    for (const drop of ["Model right about ponded / not ponded", "Next-dry forecast error (median)"]) {
      const i = rows.findIndex((r) => r[0] === drop);
      if (i >= 0) rows.splice(i, 1);
    }
    const st = R.replay_stats, C = R.campaign;
    rows.unshift(["Scored against", `${st.logger_field_days} logger field-days (${R.loggers.length} loggers)`],
      ["Ponded or not, vs loggers", st.ponded_bacc === null ? "–" : `${(st.ponded_bacc * 100).toFixed(0)} % (balanced)`],
      ["Next-dry error vs loggers (median)", st.next_dry_err_days === null ? "–" : `${st.next_dry_err_days.toFixed(2)} days (${st.forecasts_scored} forecasts; ${st.forecasts_censored} censored by irrigation or rain)`]);
    rows.push(["Farmers using the app", `${C.farmers_using_app} of ${R.n} · ${C.logged_irrigations} irrigations logged`]);
    const qcBad = R.qc.loggers.filter((l) => l.status === "bad" || l.status === "warn");
    rows.push(["Data QC (see the Data QC tab)", `${Object.entries(R.qc.counts).filter(([, v]) => v).map(([k, v]) => `${v} ${SEV[k].toLowerCase()}`).join(" · ")}` +
      (qcBad.length ? `; loggers: ${qcBad.map((l) => l.id).join(", ")}` : "")]);
  }
  const issues = replay() && R.campaign.issues.length ? `<details class="issues"><summary>Data notes (${R.campaign.issues.length})</summary><ul>${R.campaign.issues.map((x) => `<li>${x}</li>`).join("")}</ul></details>` : "";
  $("#season").innerHTML = `<h2>This season</h2>${issues}<table>${rows.map(([l, v]) => `<tr><td>${l}</td><td>${v}</td></tr>`).join("")}</table>`;
}

// ── data quality control (replay) ──────────────────────────────────────────
const esc = (x) => String(x ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
function qcWhere(q) { return q.logger ? `Logger ${q.logger} · ${q.field_id}` : q.field_id ? `Field ${q.field_id}` : "Network"; }
function qcPass(q) {
  if (qcFilter === "all") return true;
  if (qcFilter === "open") return q.severity !== "info" && q.start <= day && day <= q.end;
  return QC_GROUPS[qcFilter].includes(q.kind);
}
function renderQC() {
  const el = $("#qcpanel"), Q = R.qc;
  if (!Q) { el.innerHTML = "<p class='muted'>No quality checks for this run.</p>"; return; }
  if (qcDrawn !== R.run) {
    qcDrawn = R.run;
    const wk = Math.max(0, Q.weeks.findIndex((w) => w.start <= day && day <= w.end));
    el.innerHTML = `<div class="qc-head">
        ${Object.entries(Q.counts).map(([k, v]) => `<div class="kpi"><div class="l"><span class="sev sev-${k}">${SEV[k]}</span></div><div class="n">${v}</div><div class="s">issues</div></div>`).join("")}
        <div class="kpi"><div class="l">Logger readings received</div><div class="n">${Math.round(100 * Q.loggers.reduce((a, l) => a + l.coverage, 0) / Math.max(1, Q.loggers.length))} %</div><div class="s">${Q.loggers.length} loggers, in season</div></div>
        <div class="kpi"><div class="l">Satellite calls vs loggers</div><div class="n">${Math.round(100 * Q.network_sat_disagree)} %</div><div class="s">disagree (all sensors)</div></div>
      </div>
      <svg id="qctime" class="qc-time" role="img" aria-label="Data received per day; click to go to a day"></svg>
      <div class="qc-bar">
        <div class="seg" id="qcfilter">${[["open", "Open now"], ["all", "All"], ["loggers", "Loggers"], ["farmers", "Farmers"], ["network", "Network"]]
          .map(([k, l]) => `<button data-f="${k}" class="${k === qcFilter ? "on" : ""}">${l}</button>`).join("")}</div>
        <span class="qc-report">Weekly report <select id="qcweek">${Q.weeks.map((w, i) => `<option value="${i}" ${i === wk ? "selected" : ""}>${w.week} (${w.start_date.slice(5)} to ${w.end_date.slice(5)})</option>`).join("")}</select>
          <a id="qcmd" class="btnlink" download>Report (.md)</a> <a class="btnlink" href="/api/qc/${R.run}/issues.csv" download>Issues (.csv)</a></span>
      </div>
      <div class="qc-table-wrap"><table class="qc-table"><thead><tr><th>Severity</th><th>Where</th><th>Issue</th><th>Dates</th><th>What to do</th></tr></thead><tbody id="qcrows"></tbody></table></div>
      <details class="qc-week" id="qcweekbox"><summary>Report preview</summary><pre id="qcpre"></pre></details>
      <h2 class="qc-h">Loggers <span class="muted">reading (ink) and the model's median (blue); shaded: an issue</span></h2>
      <div class="qc-loggers" id="qclog"></div>`;
    el.querySelectorAll("#qcfilter button").forEach((b) => (b.onclick = () => {
      qcFilter = b.dataset.f; el.querySelectorAll("#qcfilter button").forEach((x) => x.classList.toggle("on", x === b)); qcRows();
    }));
    const setWeek = () => {
      const href = `/api/qc/${R.run}/report.md?week=${$("#qcweek").value}`;
      $("#qcmd").href = href;
      if ($("#qcweekbox").open) fetch(href).then((r) => r.text()).then((t) => ($("#qcpre").textContent = t));
    };
    $("#qcweek").onchange = setWeek;
    $("#qcweekbox").ontoggle = setWeek;
    setWeek();
    $("#qctime").onclick = (e) => {
      const r = e.currentTarget.getBoundingClientRect();
      setDay(Math.max(0, Math.min(R.T - 1, Math.floor(((e.clientX - r.left) / r.width) * R.T))));
    };
    qcLoggers();
  }
  qcTimeline();
  qcRows();
  document.querySelectorAll("#qclog .qc-card").forEach((c) => {
    const on = qcOnDay(c.dataset.id, day);
    c.classList.toggle("now-high", on.some((q) => q.severity === "high"));
    c.classList.toggle("now-medium", !on.some((q) => q.severity === "high") && on.some((q) => q.severity === "medium"));
    c.querySelector(".cursor").setAttribute("x1", (day + 0.5) * 4); c.querySelector(".cursor").setAttribute("x2", (day + 0.5) * 4);
  });
}
function qcTimeline() {
  const Q = R.qc, D = Q.days, T = R.T, svg = $("#qctime"), cw = 6, rows = 4, rh = 14, W = T * cw, H = rows * rh + 6;
  svg.setAttribute("viewBox", `0 0 ${W + 96} ${H}`);
  let s = "";
  const lab = (i, t) => (s += `<text x="${W + 6}" y="${i * rh + 11}">${t}</text>`);
  for (let d = 0; d < T; d++) {
    const e = D.loggers_expected[d], sh = e ? D.loggers_reporting[d] / e : null;
    if (sh !== null) {
      const h = Math.max(1, sh * (rh - 3));
      s += `<rect x="${d * cw}" y="${rh - 1 - h}" width="${cw - 1}" height="${h}" fill="var(--${sh < 0.6 ? "wrong" : sh < 0.9 ? "warn" : "ink2"})"><title>${fmtDate(d)}: ${D.loggers_reporting[d]} of ${e} loggers</title></rect>`;
    }
    SENSOR_ORDER.forEach((k, j) => {
      const c = D.satellite_calls[k][d];
      if (c) s += `<rect x="${d * cw}" y="${(j + 1) * rh + 2}" width="${cw - 1}" height="${rh - 5}" fill="${SENSORS[k].color}" opacity="${0.35 + 0.65 * Math.min(1, c / R.n)}"><title>${fmtDate(d)}: ${SENSORS[k].label}, ${c} fields</title></rect>`;
    });
  }
  lab(0, "loggers"); SENSOR_ORDER.forEach((k, j) => lab(j + 1, SENSORS[k].label));
  for (const q of Q.issues) if (q.severity === "high" || q.severity === "medium")
    s += `<line x1="${q.start * cw}" x2="${(q.end + 1) * cw - 1}" y1="${H - 2}" y2="${H - 2}" stroke="var(--${q.severity === "high" ? "wrong" : "warn"})" stroke-width="2"><title>${esc(q.title)}</title></line>`;
  s += `<line x1="${(day + 0.5) * cw}" x2="${(day + 0.5) * cw}" y1="0" y2="${H}" stroke="var(--accent)" stroke-width="1.5"/>`;
  svg.innerHTML = s;
}
function qcRows() {
  const list = R.qc.issues.filter(qcPass);
  $("#qcrows").innerHTML = list.length ? list.map((q) => `<tr data-i="${q.id}" class="${q.start <= day && day <= q.end ? "" : "past"}">
      <td><span class="sev sev-${q.severity}">${SEV[q.severity]}</span></td><td>${esc(qcWhere(q))}</td>
      <td><b>${esc(q.title)}</b><div class="muted">${esc(q.detail)}</div></td>
      <td class="mono">${q.start_date.slice(5)}${q.end !== q.start ? ` to ${q.end_date.slice(5)}` : ""}</td><td>${esc(q.action)}</td></tr>`).join("")
    : `<tr><td colspan="5" class="muted">No issues ${qcFilter === "open" ? `open on ${fmtDate(day)}` : "in this group"}.</td></tr>`;
  $("#qcrows").querySelectorAll("tr[data-i]").forEach((tr) => (tr.onclick = () => {
    const q = R.qc.issues[+tr.dataset.i];
    setDay(q.start);
    if (q.field !== null) select(q.field);
  }));
}
function qcLoggers() {
  const Q = R.qc, T = R.T, W = T * 4, H = 44, y = (h) => 2 + ((150 - Math.max(-250, Math.min(150, h))) / 400) * (H - 4);
  const line = (arr) => { let p = "", on = false; arr.forEach((v, d) => { if (v === null || v === undefined) { on = false; return; } p += `${on ? "L" : "M"}${(d * 4 + 2).toFixed(0)},${y(v).toFixed(1)}`; on = true; }); return p; };
  $("#qclog").innerHTML = Q.loggers.map((L) => {
    const lg = R.loggers.find((x) => x.id === L.id), iss = Q.issues.filter((q) => q.logger === L.id);
    const shade = iss.map((q) => `<rect x="${q.start * 4}" y="0" width="${(q.end - q.start + 1) * 4}" height="${H}" fill="var(--${q.severity === "high" ? "wrong" : q.severity === "medium" ? "warn" : "alert"})" opacity=".14"/>`).join("");
    const dr = L.drift, drift = dr ? `drift fit ${dr.slope > 0 ? "+" : ""}${dr.slope} mm/d (t ${dr.t})` : "";
    return `<button class="qc-card st-${L.status}" data-id="${L.id}" data-f="${L.field}" title="${esc(iss.map((q) => q.title).join("\n") || "No issues")}">
      <div class="qc-card-h"><b>${L.id}</b> <span class="muted">${L.field_id}</span><span class="dot"></span></div>
      <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${shade}<line x1="0" x2="${W}" y1="${y(0)}" y2="${y(0)}" stroke="var(--line)"/>
        <path d="${line(R.belief.q50[L.field])}" fill="none" stroke="var(--accent)" stroke-width="1.2" vector-effect="non-scaling-stroke"/>
        <path d="${line(lg.reading)}" fill="none" stroke="var(--ink)" stroke-width="1.2" vector-effect="non-scaling-stroke"/>
        <line class="cursor" x1="0" x2="0" y1="0" y2="${H}" stroke="var(--accent)" stroke-dasharray="2 2" vector-effect="non-scaling-stroke"/></svg>
      <div class="qc-card-f muted">${Math.round(L.coverage * 100)} % readings · sat ${L.sat_wrong}/${L.sat_checked} off${drift ? ` · ${drift}` : ""}</div></button>`;
  }).join("");
  $("#qclog").querySelectorAll(".qc-card").forEach((c) => (c.onclick = () => select(+c.dataset.f)));
}

// ── logger placement planner (app/planner.py) ──────────────────────────────
const ORDINAL = { "channel position": ["head", "middle", "tail"], "plot size": ["small", "medium", "large"] };
const PL = { key: null, info: null, sel: new Set(), locked: new Set(), excluded: new Set(), colorBy: null, mode: "toggle", ev: null, timer: null, loading: false };
const planKey = () => `${R.params.site || "grid"}|${R.n}`;
const planReq = (extra = {}) => ({ site: R.params.site || "grid", n_fields: R.n, fields: [...PL.sel], p_fail: +$("#pl_fail").value,
  pairs_target: +$("#pl_pairs").value, ...extra });
async function post(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(await r.text());
  return r;
}
function planReady() {
  if (PL.key === planKey() && PL.info) return true;
  if (PL.loading) return false;
  PL.loading = true;
  const key = planKey();
  post("/api/plan/site", { site: R.params.site || "grid", n_fields: R.n }).then((r) => r.json()).then((info) => {
    PL.loading = false;
    if (key !== planKey()) return;
    Object.assign(PL, { key, info, sel: new Set(info.current), locked: new Set(), excluded: new Set(), ev: null });
    PL.colorBy = ["soil", "channel position"].find((k) => info.strata[k]) || Object.keys(info.strata)[0];
    planPanel();
    planEval();
    draw();
  }).catch((e) => { PL.loading = false; $("#status").textContent = `Planner failed: ${e.message}`; });
  return false;
}
function planColor(f) {
  const k = PL.colorBy, v = PL.info.strata[k][f];
  if (ORDINAL[k]) return colors[`seq${ORDINAL[k].indexOf(v) + 1}`] || colors.mid;
  const i = PL.info.levels[k].indexOf(v);
  return i < 8 ? colors[`cat${i + 1}`] : colors.mid;
}
const unavailable = (f) => !PL.info.eligible[f] || PL.excluded.has(f);
function drawPlanCell(g, f) {
  const L = layout, pp = L.polys[f], bb = L.bbox[f];
  g.save(); tracePoly(g, pp); g.clip();
  g.fillStyle = planColor(f); g.globalAlpha = PL.sel.has(f) ? 1 : 0.55; g.fillRect(bb.x0, bb.y0, bb.x1 - bb.x0, bb.y1 - bb.y0); g.globalAlpha = 1;
  if (unavailable(f)) { g.fillStyle = "rgba(0,0,0,.25)"; g.fillRect(bb.x0, bb.y0, bb.x1 - bb.x0, bb.y1 - bb.y0); stripes(g, bb.x0, bb.y0, bb.x1 - bb.x0, bb.y1 - bb.y0); }
  g.restore();
  tracePoly(g, pp); g.strokeStyle = colors.surface; g.lineWidth = 1.2; g.stroke();
  if (PL.sel.has(f)) { tracePoly(g, pp); g.strokeStyle = PL.locked.has(f) ? colors.accent : colors.ink; g.lineWidth = 2.2; g.stroke(); }
}
function drawPlanPipes(g) {
  const P = layout.P, pipes = PL.info.pipes;
  if (PL.ev) for (const pr of PL.ev.pairs) {
    const [ax, ay] = P(...pipes[pr.a]), [bx, by] = P(...pipes[pr.b]);
    g.beginPath(); g.moveTo(ax, ay); g.lineTo(bx, by); g.strokeStyle = colors.ink; g.lineWidth = 2; g.setLineDash([4, 3]); g.stroke(); g.setLineDash([]);
  }
  for (const f of PL.sel) {
    const [x, y] = P(...pipes[f]);
    g.beginPath(); g.arc(x, y, 4.5, 0, 7); g.fillStyle = PL.locked.has(f) ? colors.accent : colors.surface; g.fill();
    g.lineWidth = 1.5; g.strokeStyle = unavailable(f) ? colors.wrong : colors.ink; g.stroke();
  }
}
function planClick(f) {
  if (PL.mode === "exclude") {
    if (PL.excluded.has(f)) PL.excluded.delete(f); else { PL.excluded.add(f); PL.sel.delete(f); PL.locked.delete(f); }
  } else if (PL.mode === "lock") {
    if (unavailable(f)) return;
    if (PL.locked.has(f)) PL.locked.delete(f); else { PL.locked.add(f); PL.sel.add(f); }
  } else {
    if (PL.sel.has(f)) { PL.sel.delete(f); PL.locked.delete(f); } else if (!unavailable(f)) PL.sel.add(f);
  }
  draw(); planEvalSoon();
}
function planTip(e) {
  const tip = $("#tip"), f = hitField(e);
  if (f === null || !PL.info) { tip.hidden = true; return; }
  const I = PL.info, adj = (I.adjacent[f] || []).length;
  tip.innerHTML = `<b>Plot ${R.layout.fields[f].id}</b> · ${(R.layout.fields[f].area_m2 / 1e4).toFixed(2)} ha<br>` +
    Object.keys(I.strata).map((k) => `${k}: ${I.strata[k][f]}`).join("<br>") +
    `<br>farmer consent: ${I.consent[f] || "not recorded"}${PL.excluded.has(f) ? " · <b>excluded</b>" : ""}<br>${adj} adjacent plot${adj === 1 ? "" : "s"}` +
    (PL.sel.has(f) ? `<br><b>${PL.locked.has(f) ? "In the plan (locked)" : "In the plan"}</b>` : "");
  const rect = $("#map").getBoundingClientRect();
  tip.hidden = false;
  tip.style.left = `${Math.min(e.clientX - rect.left + 14, rect.width - tip.offsetWidth - 4)}px`;
  tip.style.top = `${e.clientY - rect.top + 14}px`;
}
function planLegend() {
  if (!PL.info) return "";
  const k = PL.colorBy, lv = ORDINAL[k] ? ORDINAL[k].filter((x) => PL.info.levels[k].includes(x)) : PL.info.levels[k];
  return `<span>Plots by ${k}:</span>` + lv.map((v) => {
    const i = PL.info.levels[k].indexOf(v), c = ORDINAL[k] ? colors[`seq${ORDINAL[k].indexOf(v) + 1}`] : i < 8 ? colors[`cat${i + 1}`] : colors.mid;
    return `<span><span class="k" style="background:${c};border-radius:2px"></span> ${v}</span>`;
  }).join("") + `<span>· plan: dark outline and a pipe <span class="k" style="border:1.5px solid var(--ink);border-radius:50%"></span>, locked in blue ·
    dashed line: adjacent pair · hatched: no consent or excluded · click a plot to ${PL.mode === "toggle" ? "add or remove it" : PL.mode}</span>`;
}
function planPanel() {
  const I = PL.info, N = R.n;
  $("#planpanel").innerHTML = `<div class="plan-ctl">
      <label>Colour plots by <select id="pl_color">${Object.keys(I.strata).map((k) => `<option ${k === PL.colorBy ? "selected" : ""}>${k}</option>`).join("")}</select></label>
      <span class="seg" id="pl_mode"><button data-m="toggle" class="on">Add / remove</button><button data-m="lock">Lock</button><button data-m="exclude">Exclude</button></span>
      <label>Loggers failing <select id="pl_fail"><option value="0.05">5 %</option><option value="0.1" selected>10 %</option><option value="0.2">20 %</option><option value="0.3">30 %</option></select></label>
      <label>Adjacent pairs wanted <input id="pl_pairs" type="number" min="0" max="30" value="10"></label>
    </div>
    <div class="plan-ctl">
      <label>Plan size <input id="pl_n" type="number" min="5" max="${N}" value="${Math.min(40, Math.floor(N / 2))}"></label>
      <button id="pl_suggest" class="primary">Suggest a plan</button>
      <button id="pl_current" title="The loggers in the site file">Start from the site's loggers</button>
      <button id="pl_clear">Clear</button>
      <button id="pl_csv">Installation list (.csv)</button>
      <button id="pl_geo" ${(R.params.site || "grid") === "grid" ? "disabled title='The generated grid has no site file'" : ""}>Site file with the plan (.geojson)</button>
    </div>
    <div class="kpis" id="pl_kpis"></div><div class="plan-warn" id="pl_warn"></div>
    <h2>Coverage <span class="muted" style="text-transform:none;letter-spacing:0;font-weight:400">share of plots (grey) and of the plan's loggers (blue) in each stratum</span></h2>
    <div class="plan-strata" id="pl_strata"></div>`;
  $("#pl_color").onchange = (e) => { PL.colorBy = e.target.value; draw(); legend(); };
  document.querySelectorAll("#pl_mode button").forEach((b) => (b.onclick = () => {
    PL.mode = b.dataset.m; document.querySelectorAll("#pl_mode button").forEach((x) => x.classList.toggle("on", x === b)); legend();
  }));
  $("#pl_fail").onchange = $("#pl_pairs").onchange = () => planEval();
  $("#pl_current").onclick = () => { PL.sel = new Set([...PL.info.current, ...PL.locked]); draw(); planEval(); };
  $("#pl_clear").onclick = () => { PL.sel = new Set(PL.locked); draw(); planEval(); };
  $("#pl_suggest").onclick = async () => {
    const b = $("#pl_suggest"); b.disabled = true; b.textContent = "Searching…";
    try {
      PL.ev = await (await post("/api/plan/suggest", planReq({ n: +$("#pl_n").value, locked: [...PL.locked], excluded: [...PL.excluded] }))).json();
      PL.sel = new Set(PL.ev.fields); draw(); planMetrics();
    } catch (e) { $("#pl_warn").innerHTML = `<div>Suggest failed: ${esc(e.message)}</div>`; }
    b.disabled = false; b.textContent = "Suggest a plan";
  };
  const dl = async (fmt, name) => {
    try {
      const blob = await (await post("/api/plan/export", planReq({ format: fmt }))).blob();
      const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = name; a.click(); URL.revokeObjectURL(a.href);
    } catch (e) { $("#pl_warn").innerHTML = `<div>Export failed: ${esc(e.message)}</div>`; }
  };
  $("#pl_csv").onclick = () => dl("csv", `logger_plan_${R.params.site || "grid"}.csv`);
  $("#pl_geo").onclick = () => dl("geojson", `${R.params.site}_plan.geojson`);
}
function planEvalSoon() { clearTimeout(PL.timer); PL.timer = setTimeout(planEval, 150); }
async function planEval() {
  if (!PL.info) return;
  try { PL.ev = await (await post("/api/plan/evaluate", planReq())).json(); } catch (e) { $("#pl_warn").innerHTML = `<div>${esc(e.message)}</div>`; return; }
  draw(); planMetrics();
}
function planMetrics() {
  const E = PL.ev, C = E.calibration, x = C.exp19, pct0 = (v) => `${Math.round(v * 100)} %`;
  const unav = PL.info.eligible.filter((v) => !v).length + [...PL.excluded].filter((f) => PL.info.eligible[f]).length;
  const items = [
    ["Loggers in the plan", `${E.n}`, `${PL.locked.size} locked · ${unav} plots unavailable`],
    [`At least ${E.need} still working`, pct0(C.p_enough), `at ${pct0(E.p_fail)} failing, ${C.expected_working} expected; ${C.plan_size_for["0.9"]} / ${C.plan_size_for["0.95"]} loggers give 90 / 95 %`],
    ["Interval coverage (Exp. 19)", x ? `${x.sp.toFixed(2)} · ${x.fc.toFixed(2)}` : "–", x ? `S&P · forecast, target 0.90; 1 campaign in 10 gets ≤ ${x.sp_p10.toFixed(2)} · ${x.fc_p10.toFixed(2)}` : "under 3 working loggers"],
    ["Balance across strata", E.n ? pct0(E.balance_mean) : "–", E.balance_vs_random !== null ? `better than ${pct0(E.balance_vs_random)} of random plans (their median ${pct0(E.random_balance_median)})` : ""],
    ["Adjacent pairs", `${E.pairs.length}`, `wanted ${E.pairs_target} · random plans give ${E.random_pairs_median ?? "–"} · ${E.available_pairs} on the site`],
    ["Read-out round", E.n ? `${(E.route.length_m / 1000).toFixed(1)} km · ${E.route.hours} h` : "–", E.n ? `walk at 4 km/h + 10 min per logger; a plot is a median ${E.spread.median_m} m from a logger` : ""],
  ];
  $("#pl_kpis").innerHTML = items.map(([l, n, s_]) => `<div class="kpi"><div class="l">${l}</div><div class="n">${n}</div><div class="s">${s_}</div></div>`).join("");
  const w = [];
  if (E.n < E.need) w.push(`Fewer than ${E.need} loggers: Experiment 19 found the S&P and forecast intervals too narrow, and down to luck, below about 20.`);
  else if (C.p_enough < 0.9) w.push(`Only a ${pct0(C.p_enough)} chance that ${E.need} loggers survive the season at ${pct0(E.p_fail)} failing; plan ${C.plan_size_for["0.9"]} or more.`);
  if (E.ineligible.length) w.push(`In the plan without the farmer's consent: ${E.ineligible.join(", ")}.`);
  if (E.missing.length) w.push(`No logger in: ${E.missing.join("; ")}.`);
  if (E.pairs.length < E.pairs_target) w.push(`${E.pairs.length} adjacent pairs; ${E.pairs_target} wanted for the water moving between plots.`);
  if (E.spread.far_plots) w.push(`${E.spread.far_plots} plots are more than 150 m from any logger.`);
  $("#pl_warn").innerHTML = w.length ? w.map((t) => `<div>${esc(t)}</div>`).join("") : `<div class="ok">The plan meets every check.</div>`;
  $("#pl_strata").innerHTML = E.strata.map((S) => {
    const tot = S.levels.reduce((a, l) => a + l.plots, 0), lv = ORDINAL[S.name] ? ORDINAL[S.name].map((o) => S.levels.find((l) => l.level === o)).filter(Boolean) : S.levels;
    return `<div class="plan-dim"><h3><span>${S.name}</span><span class="muted">balance ${pct0(E.balance[S.name] || 0)}</span></h3>` + lv.map((l) => {
      const i = PL.info.levels[S.name].indexOf(l.level), sw = ORDINAL[S.name] ? colors[`seq${ORDINAL[S.name].indexOf(l.level) + 1}`] : i < 8 ? colors[`cat${i + 1}`] : colors.mid;
      const miss = l.loggers === 0 && E.missing.includes(`${S.name}: ${l.level}`);
      return `<div class="plan-row" title="${esc(l.level)}: ${l.plots} plots, ${l.loggers} loggers (${l.expected} for a proportional share)">
        <span class="lv ${miss ? "miss" : ""}"><span class="sw" style="background:${sw}"></span>${esc(l.level)}</span>
        <span class="plan-bars"><span class="pf" style="width:${(100 * l.plots) / tot}%"></span><span class="pl" style="width:${E.n ? (100 * l.loggers) / E.n : 0}%"></span></span>
        <span class="num">${l.plots} plots · ${l.loggers} loggers (proportional ${l.expected})</span></div>`;
    }).join("") + "</div>";
  }).join("");
}

// ── advice against practice (src/awdrt/counterfactual.py) ─────────────────
const CF_ARMS = { model: "KGML-Paddy", regional: "Regional S&P", et_only: "ET only", perfect: "Perfect knowledge" };
const CF = { data: null, drawn: null, arm: "model", metric: "water", busy: false };
const cfArmColor = (a) => ({ practice: colors.ink, model: colors.cat1, regional: colors.cat2, et_only: colors.cat3, perfect: colors.ink2 })[a];
const cfDiff = (f) => {
  const A = CF.data.arms, k = CF.metric === "water" ? "water_mm" : "stress_days";
  return A[CF.arm].field[k][f] - A.practice.field[k][f];
};
function drawCfCell(g, f) {
  const L = layout, pp = L.polys[f], bb = L.bbox[f];
  let col = colors.mid;
  if (CF.data) {   // drier outcome (less water, more stress) brown; wetter blue
    const v = cfDiff(f), m = CF.metric === "water" ? 300 : 15, t = Math.max(-1, Math.min(1, v / m));
    const drier = CF.metric === "water" ? t < 0 : t > 0;
    col = mix(colors.mid, drier ? colors.dry : colors.wet, Math.sqrt(Math.abs(t)));
  }
  g.save(); tracePoly(g, pp); g.clip(); g.fillStyle = col; g.fillRect(bb.x0, bb.y0, bb.x1 - bb.x0, bb.y1 - bb.y0); g.restore();
  tracePoly(g, pp); g.strokeStyle = colors.surface; g.lineWidth = 1.2; g.stroke();
  if (CF.data && !CF.data.advised[f] && CF.arm !== "practice") { const [cx, cy] = L.cents[f]; g.fillStyle = colors.ink2; g.font = "10px system-ui"; g.textAlign = "center"; g.fillText("–", cx, cy + 3); }
  if (f === sel) { tracePoly(g, pp); g.strokeStyle = colors.accent; g.lineWidth = 3; g.stroke(); }
}
function cfLegend() {
  if (!CF.data) return `<span>Run the counterfactual to colour the fields.</span>`;
  const w = CF.metric === "water";
  return `<span>${CF_ARMS[CF.arm]} advice minus the farmer's practice, ${w ? "irrigation water (mm)" : "stress days (level below −15 cm)"}:</span>
    <span><div class="bar" style="background:linear-gradient(90deg,${w ? colors.dry : colors.wet},${colors.mid},${w ? colors.wet : colors.dry})"></div>
    <div class="ticks"><span>${w ? "−300 mm" : "−15 d"}</span><span>same</span><span>${w ? "+300 mm" : "+15 d"}</span></div></span>
    <span>brown: drier than practice · blue: wetter · “–”: this farmer does not follow the advice</span>`;
}
function cfTip(e) {
  const tip = $("#tip"), f = hitField(e);
  if (f === null || !CF.data) { tip.hidden = true; return; }
  const A = CF.data.arms, P = A.practice.field, M = A[CF.arm].field;
  tip.innerHTML = `<b>Field ${R.layout.fields[f].id}</b> · ${CF.data.ftype[f]} farmer${CF.data.advised[f] ? "" : " (not following)"}<br>
    Water: practice ${P.water_mm[f]} mm · ${CF_ARMS[CF.arm]} ${M.water_mm[f]} mm<br>
    Irrigations: ${P.irrigations[f]} · ${M.irrigations[f]}<br>Stress days: ${P.stress_days[f]} · ${M.stress_days[f]}` +
    (A[CF.arm].asked_days[f].length ? `<br>Asked about the water ${A[CF.arm].asked_days[f].length} times` : "");
  const rect = $("#map").getBoundingClientRect();
  tip.hidden = false;
  tip.style.left = `${Math.min(e.clientX - rect.left + 14, rect.width - tip.offsetWidth - 4)}px`;
  tip.style.top = `${e.clientY - rect.top + 14}px`;
}
function cfPanel() {
  const el = $("#cfpanel");
  if (CF.drawn !== R.run) {
    CF.drawn = R.run;
    el.innerHTML = `<div class="plan-ctl">
        <label>Advice <select id="cf_lead"><option value="1">irrigate by tomorrow</option><option value="3">book the pump 3 days ahead</option><option value="5">5 days ahead</option></select></label>
        <label>Irrigate <select id="cf_wait"><option value="0">when the water is gone</option><option value="1">one dry day later</option><option value="2">two dry days later</option></select></label>
        <label>Act at <select id="cf_risk"><option value="0.5">50 % chance of dry</option><option value="0.25">25 % (cautious)</option><option value="0.1">10 % (very cautious)</option></select></label>
      </div>
      <div class="plan-ctl">
        <label><input type="checkbox" id="cf_ask" checked> Ask the farmer when unsure</label>
        <label>Farmers following <select id="cf_follow"><option value="1">all</option><option value="0.75">75 %</option><option value="0.5">50 %</option></select></label>
        <button id="cf_run" class="primary">Run the counterfactual</button>
        <label>Pump <input id="cf_q" type="number" value="100" min="10" max="1000" step="10"> m³/h</label>
        <label><input id="cf_cost" type="number" value="80" min="0" max="10000" step="5"> Tk per pump hour</label>
      </div>
      <div id="cf_body"><p class="note">Re-runs this village from transplanting, five times: as the farmers irrigated, and following the
        advice of KGML-Paddy, of the same model with one regional leak rate, with evaporation only, and of perfect knowledge. The fields,
        weather, satellite noise and irrigation depths are the same in every run; only the timing of irrigation differs. About 10–20 s.</p></div>`;
    $("#cf_run").onclick = cfRun;
    $("#cf_q").onchange = $("#cf_cost").onchange = () => CF.data && cfBody();
  } else if (CF.data && $("#cf_cum")) { cfFieldChart(); cfCum(); }
}
async function cfRun() {
  if (CF.busy) return;
  CF.busy = true;
  const b = $("#cf_run"); b.disabled = true; b.textContent = "Running five seasons…";
  try {
    CF.data = await (await post("/api/counterfactual", { run: R.run, lead: +$("#cf_lead").value, wait: +$("#cf_wait").value,
      risk: +$("#cf_risk").value, follow: +$("#cf_follow").value, ask: $("#cf_ask").checked })).json();
    cfBody(); draw(); legend();
  } catch (e) { $("#cf_body").innerHTML = `<p class="note">Failed: ${esc(e.message)}</p>`; }
  CF.busy = false; b.disabled = false; b.textContent = "Run the counterfactual";
}
function cfBody() {
  const D = CF.data, A = D.arms, P = A.practice, M = A[CF.arm], q = +$("#cf_q").value || 100, cost = +$("#cf_cost").value || 0;
  const pct = (a, b) => (b ? `${a >= b ? "+" : "−"}${Math.abs(Math.round((100 * (a - b)) / b))} %` : "–");
  const cls = (good) => (good ? "better" : "worse");
  const dm3 = M.total_m3 - P.total_m3, dh = dm3 / q;
  const S = D.settings;
  const items = [
    ["Water pumped", `${Math.round(M.total_m3).toLocaleString()} m³`, `<span class="${cls(dm3 <= 0)}">${pct(M.total_m3, P.total_m3)}</span> against practice (${Math.round(P.total_m3).toLocaleString()} m³)`],
    ["Pump hours · cost", `${dh <= 0 ? "−" : "+"}${Math.abs(Math.round(dh)).toLocaleString()} h`, `${dh <= 0 ? "saves" : "costs"} ${Math.abs(Math.round(dh * cost)).toLocaleString()} Tk at ${q} m³/h and ${cost} Tk/h (your assumptions)`],
    ["Stress days per field", `${P.stress_days.toFixed(1)} → ${M.stress_days.toFixed(1)}`, `<span class="${cls(M.stress_days <= P.stress_days)}">${M.stress_days <= P.stress_days ? "fewer" : "more"}</span> days with the water 15 cm or more below the surface`],
    ["Irrigations per field", `${P.irrigations.toFixed(1)} → ${M.irrigations.toFixed(1)}`, `${M.water_mm} mm per field (practice ${P.water_mm} mm), after puddling`],
    ["Days ponded per field", `${P.ponded_days.toFixed(0)} → ${M.ponded_days.toFixed(0)}`, "ponded days drive methane"],
    ["Methane (IPCC Tier 1)", `${P.methane.kg_ha.toFixed(0)} → ${M.methane.kg_ha.toFixed(0)} kg/ha`, `<span class="${cls(M.methane.kg_ha <= P.methane.kg_ha)}">${pct(M.methane.kg_ha, P.methane.kg_ha)}</span>; continuously flooded fields ${P.methane.counts.continuous} → ${M.methane.counts.continuous}`],
    ["Questions per farmer", `${M.asked.toFixed(0)}`, S.ask && CF.arm !== "perfect" ? "\"how much water now?\", only when the answer could change the advice" : "no questions asked"],
  ];
  $("#cf_body").innerHTML = `<div class="seg" id="cf_arm" style="margin-bottom:8px">${Object.entries(CF_ARMS).map(([k, l]) => `<button data-a="${k}" class="${k === CF.arm ? "on" : ""}">${l}</button>`).join("")}</div>
    <span class="seg" id="cf_metric" style="margin-left:8px"><button data-m="water" class="${CF.metric === "water" ? "on" : ""}">Map: water</button><button data-m="stress" class="${CF.metric === "stress" ? "on" : ""}">Map: stress days</button></span>
    <div class="kpis">${items.map(([l, n, s_]) => `<div class="kpi"><div class="l">${l}</div><div class="n">${n}</div><div class="s">${s_}</div></div>`).join("")}</div>
    <div class="cf-grid">
      <div><h3>Water against dry stress, per field (means)</h3><svg id="cf_scatter" viewBox="0 0 420 250" role="img" aria-label="Water and stress days of each policy"></svg>
        <div class="cf-legend">${Object.keys(CF_ARMS).map((a) => `<span><i style="background:${cfArmColor(a)};height:8px;width:8px;border-radius:50%;vertical-align:0"></i>${CF_ARMS[a]}</span>`).join("")}<span>★ farmers' practice</span></div>
        <p class="note">Down and left is better. The ringed point is the advice compared on this page; hover a point for its numbers. Perfect knowledge shows what any forecast could reach.</p></div>
      <div><h3>Water pumped through the season (village, m³)</h3><svg id="cf_cum" viewBox="0 0 420 250" role="img" aria-label="Cumulative water by policy"></svg>
        <div class="cf-legend">${["practice", ...Object.keys(CF_ARMS)].map((a) => `<span><i style="background:${cfArmColor(a)}"></i>${a === "practice" ? "Farmers' practice" : CF_ARMS[a]}</span>`).join("")}</div></div>
      <div><h3>By farmer type: practice → ${CF_ARMS[CF.arm]}</h3><table><tr><th>Farmers</th><th>Fields</th><th>Water (mm)</th><th>Stress days</th></tr>
        ${Object.entries(P.by_type).map(([t, v]) => { const m = M.by_type[t]; return `<tr><td>${t}</td><td>${v.fields}</td>
          <td>${v.water_mm} → <span class="${cls(m.water_mm <= v.water_mm)}">${m.water_mm}</span></td>
          <td>${v.stress_days} → <span class="${cls(m.stress_days <= v.stress_days)}">${m.stress_days}</span></td></tr>`; }).join("")}</table>
        <p class="note">Settings: ${S.lead === 1 ? "irrigate by tomorrow" : `fixed ${S.lead} days ahead`}, ${S.wait ? `${S.wait} dry day${S.wait > 1 ? "s" : ""} after the water is gone` : "when the water is gone"},
        acting at a ${Math.round(S.risk * 100)} % chance, ${S.ask ? "asking when unsure" : "no questions"}, ${Math.round(S.follow * 100)} % of farmers following. Computed in ${D.seconds} s.
        Every number is simulation (simulator v2); the practice is the simulator's farmer rules.</p></div>
      <div><h3 id="cf_fh">Click a field to compare its water level</h3><svg id="cf_field" viewBox="0 0 420 220" role="img" aria-label="Water level of the selected field under practice and advice"></svg></div>
    </div>`;
  document.querySelectorAll("#cf_arm button").forEach((b) => (b.onclick = () => { CF.arm = b.dataset.a; cfBody(); draw(); legend(); }));
  document.querySelectorAll("#cf_metric button").forEach((b) => (b.onclick = () => { CF.metric = b.dataset.m; cfBody(); draw(); legend(); }));
  cfScatter(); cfCum(); cfFieldChart();
}
function cfScatter() {
  const A = CF.data.arms, keys = ["practice", ...Object.keys(CF_ARMS)], W = 420, H = 250, m = { l: 44, r: 16, t: 14, b: 34 };
  const xs = keys.map((k) => A[k].water_mm), ys = keys.map((k) => A[k].stress_days);
  const x0 = Math.min(...xs) * 0.95, x1 = Math.max(...xs) * 1.05, y1 = Math.max(...ys) * 1.15 + 0.5;
  const X = (v) => m.l + ((W - m.l - m.r) * (v - x0)) / (x1 - x0), Y = (v) => m.t + ((H - m.t - m.b) * (y1 - v)) / y1;
  let s = `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(0)}" y2="${Y(0)}" stroke="${colors.line}"/><line x1="${m.l}" x2="${m.l}" y1="${m.t}" y2="${Y(0)}" stroke="${colors.line}"/>`;
  for (let i = 0; i <= 4; i++) { const v = (y1 * i) / 4; s += `<text x="${m.l - 6}" y="${Y(v) + 3}" text-anchor="end">${v.toFixed(0)}</text>`; }
  for (let i = 0; i <= 4; i++) { const v = x0 + ((x1 - x0) * i) / 4; s += `<text x="${X(v)}" y="${H - m.b + 14}" text-anchor="middle">${v.toFixed(0)}</text>`; }
  s += `<text x="${(m.l + W - m.r) / 2}" y="${H - 4}" text-anchor="middle">irrigation water per field (mm)</text><text x="10" y="${m.t + 4}" transform="rotate(-90 10 ${m.t + 4})" text-anchor="end">stress days per field</text>`;
  keys.forEach((k) => {
    const x = X(A[k].water_mm), y = Y(A[k].stress_days), c = cfArmColor(k), on = k === CF.arm;
    s += k === "practice" ? `<path d="M${x},${y - 7} L${x + 2},${y - 2} L${x + 7},${y - 2} L${x + 3},${y + 1} L${x + 5},${y + 6} L${x},${y + 3} L${x - 5},${y + 6} L${x - 3},${y + 1} L${x - 7},${y - 2} L${x - 2},${y - 2} Z" fill="${c}"/>`
      : `<circle cx="${x}" cy="${y}" r="${on ? 6.5 : 5}" fill="${c}" stroke="${colors.surface}" stroke-width="2"/>${on ? `<circle cx="${x}" cy="${y}" r="9.5" fill="none" stroke="${c}" stroke-width="1.5"/>` : ""}`;
    if (k === "practice" || on) s += `<text class="lab" x="${x + 12}" y="${y - 8}">${k === "practice" ? "practice" : CF_ARMS[k]}</text>`;
    s += `<circle cx="${x}" cy="${y}" r="12" fill="transparent"><title>${k === "practice" ? "Farmers' practice" : CF_ARMS[k]}: ${A[k].water_mm} mm, ${A[k].stress_days} stress days, ${A[k].irrigations} irrigations per field</title></circle>`;
  });
  $("#cf_scatter").innerHTML = s;
}
function cfCum() {
  const A = CF.data.arms, keys = ["practice", ...Object.keys(CF_ARMS)], W = 420, H = 250, m = { l: 52, r: 12, t: 12, b: 30 }, T = R.T;
  const ymax = Math.max(...keys.map((k) => A[k].cum_m3[T - 1])) * 1.05;
  const X = (d) => m.l + ((W - m.l - m.r) * d) / (T - 1), Y = (v) => m.t + ((H - m.t - m.b) * (ymax - v)) / ymax;
  let s = `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(0)}" y2="${Y(0)}" stroke="${colors.line}"/>`;
  for (let i = 0; i <= 4; i++) { const v = (ymax * i) / 4; s += `<text x="${m.l - 6}" y="${Y(v) + 3}" text-anchor="end">${Math.round(v / 1000)}k</text>`; }
  for (let d = 0; d < T - 10; d += 30) s += `<text x="${X(d)}" y="${H - m.b + 14}" text-anchor="${d ? "middle" : "start"}">${fmtDate(d)}</text>`;
  keys.forEach((k) => {
    s += `<path d="${A[k].cum_m3.map((v, d) => `${d ? "L" : "M"}${X(d).toFixed(1)},${Y(v).toFixed(1)}`).join("")}" fill="none" stroke="${cfArmColor(k)}" stroke-width="${k === CF.arm || k === "practice" ? 2.2 : 1.3}" ${k === "perfect" ? 'stroke-dasharray="4 3"' : ""}/>`;
  });
  s += `<line x1="${X(day)}" x2="${X(day)}" y1="${m.t}" y2="${Y(0)}" stroke="${colors.accent}" stroke-dasharray="2 2"/>`;
  $("#cf_cum").innerHTML = s;
}
function cfFieldChart() {
  const svg = $("#cf_field");
  if (!svg || !CF.data) return;
  if (sel === null) { svg.innerHTML = ""; return; }
  const A = CF.data.arms, f = sel, W = 420, H = 220, m = { l: 36, r: 10, t: 12, b: 28 }, T = R.T, ymin = -250, ymax = 150;
  $("#cf_fh").textContent = `Field ${R.layout.fields[f].id} (${CF.data.ftype[f]} farmer${CF.data.advised[f] ? "" : ", not following"}): practice and ${CF_ARMS[CF.arm]}`;
  const X = (d) => m.l + ((W - m.l - m.r) * d) / (T - 1), Y = (h) => m.t + ((H - m.t - m.b) * (ymax - Math.max(ymin, Math.min(ymax, h)))) / (ymax - ymin);
  const line = (arr) => { let p = "", on = false; arr.forEach((v, d) => { if (v === null) { on = false; return; } p += `${on ? "L" : "M"}${X(d).toFixed(1)},${Y(v).toFixed(1)}`; on = true; }); return p; };
  let s = `<rect x="${m.l}" y="${Y(-150)}" width="${W - m.l - m.r}" height="${Y(ymin) - Y(-150)}" fill="${colors.dry}" opacity=".08"/>`;
  for (const v of [-200, -100, 0, 100]) s += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="${colors.line}"/><text x="${m.l - 5}" y="${Y(v) + 3}" text-anchor="end">${v}</text>`;
  for (let d = 0; d < T - 10; d += 30) s += `<text x="${X(d)}" y="${H - m.b + 14}" text-anchor="${d ? "middle" : "start"}">${fmtDate(d)}</text>`;
  s += `<text x="${m.l + 4}" y="${Y(-150) + 11}">stress: 15 cm or more below the surface</text>`;
  for (const [k, dy] of [["practice", 0], [CF.arm, 6]]) {
    s += `<path d="${line(A[k].H[f])}" fill="none" stroke="${cfArmColor(k)}" stroke-width="1.6"/>`;
    A[k].irr[f].forEach((v, d) => { if (v > 0) s += `<path d="M${X(d)},${H - m.b - 2 - dy} l3,-5 l-6,0 z" fill="${cfArmColor(k)}"><title>${k === "practice" ? "practice" : CF_ARMS[k]}: ${v} mm on ${fmtDate(d)}</title></path>`; });
  }
  A[CF.arm].asked_days[f].forEach((d) => (s += `<circle cx="${X(d)}" cy="${m.t + 3}" r="2.2" fill="${cfArmColor(CF.arm)}"><title>asked on ${fmtDate(d)}</title></circle>`));
  s += `<line x1="${X(day)}" x2="${X(day)}" y1="${m.t}" y2="${H - m.b}" stroke="${colors.accent}" stroke-dasharray="2 2"/>`;
  svg.innerHTML = s;
}

// ── methane (src/awdrt/methane.py) ──────────────────────────────────────────
const MH = { data: null, drawn: null, src: "truth", busy: false };
const MH_SRC = () => ({ truth: replay() ? "Loggers" : "True water level", model: "KGML-Paddy belief", satellite: "Satellite calls only", regional: "Regional default" });
const MH_REG = { continuous: "continuously flooded", single: "single drainage", multiple: "multiple drainage (AWD)", aggregated: "IPCC aggregated (regime unknown)" };
function mhColor(kg) {   // one hue, light to dark, over the season's usual range: 25 to 70 kg CH4/ha
  if (kg === null || kg === undefined) return colors.line;
  return mix(colors.mid, colors.warn, Math.max(0, Math.min(1, (kg - 25) / 45)));
}
function drawMhCell(g, f) {
  const L = layout, pp = L.polys[f], bb = L.bbox[f];
  const S = MH.data ? MH.data.sources[MH.src] : null, kg = S ? S.kg_ha[f] : null;
  g.save(); tracePoly(g, pp); g.clip(); g.fillStyle = S ? mhColor(kg) : colors.mid; g.fillRect(bb.x0, bb.y0, bb.x1 - bb.x0, bb.y1 - bb.y0);
  if (S && kg === null) stripes(g, bb.x0, bb.y0, bb.x1 - bb.x0, bb.y1 - bb.y0);
  g.restore();
  tracePoly(g, pp); g.strokeStyle = colors.surface; g.lineWidth = 1.2; g.stroke();
  if (S && S.regime[f] === "continuous") { const [cx, cy] = L.cents[f]; g.fillStyle = colors.ink; g.font = "700 10px system-ui"; g.textAlign = "center"; g.textBaseline = "middle"; g.fillText("CF", cx, cy); }
  if (MH.data && MH.src !== "truth" && MH.src !== "regional" && MH.data.truth_fields[f] && S.regime[f] !== MH.data.sources.truth.regime[f]) {
    tracePoly(g, pp); g.strokeStyle = colors.wrong; g.setLineDash([4, 3]); g.lineWidth = 2; g.stroke(); g.setLineDash([]);
  }
  if (f === sel) { tracePoly(g, pp); g.strokeStyle = colors.accent; g.lineWidth = 3; g.stroke(); }
}
function mhLegend() {
  if (!MH.data) return "<span>Loading…</span>";
  return `<span>${MH_SRC()[MH.src]}: methane, kg CH₄/ha this season (IPCC Tier 1):</span>
    <span><div class="bar" style="background:linear-gradient(90deg,${mhColor(25)},${mhColor(47)},${mhColor(70)})"></div><div class="ticks"><span>≤ 25</span><span>47</span><span>≥ 70</span></div></span>
    <span><b>CF</b> continuously flooded${MH.src === "model" || MH.src === "satellite" ? ` · <span class="k k-wrong"></span> regime differs from the ${replay() ? "logger" : "truth"}` : ""}${replay() ? " · hatched: no logger" : ""}</span>`;
}
function mhTip(e) {
  const tip = $("#tip"), f = hitField(e);
  if (f === null || !MH.data) { tip.hidden = true; return; }
  const D = MH.data;
  tip.innerHTML = `<b>Field ${R.layout.fields[f].id}</b> · ${D.days[f]} days in season<br>` +
    Object.entries(MH_SRC()).map(([k, l]) => { const S = D.sources[k]; return S.kg_ha[f] === null ? `${l}: –` :
      `${l}: ${MH_REG[S.regime[f]]}${S.periods ? ` (${S.periods[f]} drainage period${S.periods[f] === 1 ? "" : "s"})` : ""}, <b>${S.kg_ha[f]}</b> kg/ha`; }).join("<br>");
  const rect = $("#map").getBoundingClientRect();
  tip.hidden = false;
  tip.style.left = `${Math.min(e.clientX - rect.left + 14, rect.width - tip.offsetWidth - 4)}px`;
  tip.style.top = `${e.clientY - rect.top + 14}px`;
}
function mhPanel() {
  const el = $("#mhpanel");
  if (MH.drawn !== R.run) {
    MH.drawn = R.run;
    el.innerHTML = `<div class="plan-ctl">
        <span class="seg" id="mh_src">${Object.entries(MH_SRC()).map(([k, l]) => `<button data-s="${k}" class="${k === MH.src ? "on" : ""}">${l}</button>`).join("")}</span>
      </div>
      <div class="plan-ctl">
        <label>A drainage period is at least <select id="mh_min">${[1, 2, 3, 5].map((v) => `<option value="${v}" ${v === 2 ? "selected" : ""}>${v} day${v > 1 ? "s" : ""}</option>`).join("")}</select> without standing water</label>
        <label>Regional default <select id="mh_reg"><option value="aggregated">IPCC "irrigated", regime unknown (0.60)</option><option value="continuous">continuously flooded (1.00)</option><option value="multiple">multiple drainage (0.55)</option></select></label>
      </div><div id="mh_body"><p class="note">Computing…</p></div>`;
    el.querySelectorAll("#mh_src button").forEach((b) => (b.onclick = () => {
      MH.src = b.dataset.s; el.querySelectorAll("#mh_src button").forEach((x) => x.classList.toggle("on", x === b)); draw(); legend();
    }));
    $("#mh_min").onchange = $("#mh_reg").onchange = mhLoad;
    mhLoad();
  }
}
async function mhLoad() {
  if (MH.busy) return;
  MH.busy = true;
  try {
    MH.data = await (await post("/api/methane", { run: R.run, min_dry: +$("#mh_min").value, regional: $("#mh_reg").value })).json();
    mhBody(); draw(); legend();
  } catch (e) { $("#mh_body").innerHTML = `<p class="note">Failed: ${esc(e.message)}</p>`; }
  MH.busy = false;
}
function mhBody() {
  const D = MH.data, S = D.sources, T = S.truth, ha = R.layout.fields.reduce((a, f, i) => a + (D.truth_fields[i] ? f.area_m2 : 0), 0) / 1e4;
  const t = (kg) => (kg * D.factors.gwp100 / 1000).toFixed(1);
  const rows = Object.entries(MH_SRC()).map(([k, l]) => {
    const s_ = S[k], v = s_.vs_truth;
    return `<tr><td>${l}</td><td>${s_.counts.continuous}</td><td>${s_.counts.single}</td><td>${s_.counts.multiple}</td><td>${s_.counts.aggregated || "–"}</td>
      <td>${Math.round(s_.total_kg).toLocaleString()} <span class="muted">(${Math.round(s_.total_lo).toLocaleString()}–${Math.round(s_.total_hi).toLocaleString()})</span></td>
      <td>${t(s_.total_kg)}</td>
      <td>${v ? `${v.total_err >= 0 ? "+" : ""}${Math.round(v.total_err * 100)} %` : "–"}</td>
      <td>${v && v.regime_balanced !== null ? `${Math.round(v.regime_balanced * 100)} %` : "–"}</td>
      <td>${v ? `${Math.round(v.half_gap_share * 100)} %` : "–"}</td></tr>`;
  }).join("");
  const C = S.model.vs_truth.confusion, regs = ["continuous", "single", "multiple"];
  const conf = `<table><tr><th>${replay() ? "Logger" : "True"} regime ↓ · model →</th>${regs.map((r) => `<th>${MH_REG[r].split(" ")[0]}</th>`).join("")}</tr>
    ${regs.map((a) => `<tr><td>${MH_REG[a]}</td>${regs.map((b) => `<td>${a === b ? `<b>${C[a][b]}</b>` : C[a][b]}</td>`).join("")}</tr>`).join("")}</table>`;
  $("#mh_body").innerHTML = `<div class="kpis">
      <div class="kpi"><div class="l">Methane, ${replay() ? "logger fields" : "village"} (${MH_SRC().truth.toLowerCase()})</div><div class="n">${Math.round(T.total_kg).toLocaleString()} kg</div><div class="s">${t(T.total_kg)} t CO₂e · 95 % range ${Math.round(T.total_lo).toLocaleString()}–${Math.round(T.total_hi).toLocaleString()} kg from the IPCC factors · ${ha.toFixed(1)} ha</div></div>
      <div class="kpi"><div class="l">Per hectare</div><div class="n">${(T.total_kg / ha).toFixed(0)} kg</div><div class="s">this season; continuous flooding would give ${Math.round(D.factors.ef_c[0] * D.days.reduce((a, d, i) => a + (D.truth_fields[i] ? d * R.layout.fields[i].area_m2 / 1e4 : 0), 0) / ha)} kg/ha</div></div>
      <div class="kpi"><div class="l">Regimes the model gets right (balanced)</div><div class="n">${Math.round(S.model.vs_truth.regime_balanced * 100)} %</div><div class="s">satellite only ${Math.round(S.satellite.vs_truth.regime_balanced * 100)} % · continuously flooded fields found: model ${S.model.vs_truth.cf_recall === null ? "–" : Math.round(S.model.vs_truth.cf_recall * 100) + " %"}, satellite ${S.satellite.vs_truth.cf_recall === null ? "–" : Math.round(S.satellite.vs_truth.cf_recall * 100) + " %"} · ${S.model.vs_truth.fields} ${replay() ? "logger " : ""}fields</div></div>
      <div class="kpi"><div class="l">Fields the regional default misjudges (H5b)</div><div class="n">${Math.round(S.regional.vs_truth.half_gap_share * 100)} %</div><div class="s">off by at least half the gap between continuous flooding and AWD; village total ${S.regional.vs_truth.total_err >= 0 ? "+" : ""}${Math.round(S.regional.vs_truth.total_err * 100)} %</div></div>
    </div>
    <div class="cf-grid">
      <div style="grid-column:1/-1"><h3>Water regime and methane by source</h3><table><tr><th>Source</th><th>Contin.</th><th>Single</th><th>Multiple</th><th>Aggr.</th><th>kg CH₄ (95 % range)</th><th>t CO₂e</th><th>Total vs ${replay() ? "loggers" : "truth"}</th><th>Regime right (balanced)</th><th>Fields off ≥ ½ gap</th></tr>${rows}</table>
        <p class="note">IPCC 2019 Refinement Tier 1: ${D.factors.ef_c[0]} kg CH₄/ha/day (South Asia, 95 % range ${D.factors.ef_c[1]}–${D.factors.ef_c[2]}) × water regime
        (continuously flooded 1.00, single drainage 0.71, multiple drainage 0.55, regime unknown 0.60) × days in season; no flooding before the season, no organic amendments.
        A drainage period is at least ${D.settings.min_dry} day${D.settings.min_dry > 1 ? "s" : ""} without standing water, not counting the end-of-season drainage. The 95 % range comes from the IPCC factor ranges, shared by all fields.
        GWP100 ${D.factors.gwp100} (AR6). ${replay() ? "In replay the truth is the loggers, so the comparison uses logger fields only." : "Every number is simulation."}</p></div>
      <div><h3>Which regime the model sees</h3>${conf}</div>
      <div><h3>Why it matters</h3><p class="note">A regional default can get the village total about right while misjudging individual fields,
        because the errors cancel. Field-level claims, such as a carbon-credit payment per farm for AWD, need the field's own water history:
        that is what the loggers, the model and the satellite each try to give.</p></div>
    </div>`;
}

// ── experiment mode (app/experiment.py) ────────────────────────────────────
const EX = { built: false, job: null, res: null, timer: null, sel: "ponded_bacc" };
const EX_VARY = {
  model: ["Model", [["v1b", "v1b"], ["v1c", "v1c (wide prior)"]]],
  nisar_update: ["NISAR answer update", [["false", "off"], ["true", "on"]]],
  world: ["Simulator world", [["prior", "same as training"], ["reality", "random mismatch"], ["reality_systematic", "random + systematic"]]],
  site: ["Field layout", null],
  leak: ["Soil leakiness (S&P)"], et0_scale: ["Evaporation ×"], rain_scale: ["Rain ×"], cloud: ["Extra cloud"],
  share_logging: ["Farmers using the app"], p_awd: ["Farmers doing AWD"], p_topup: ["Farmers topping up"],
  pump_bias: ["Pump rating error ×"], nisar_acc: ["NISAR accuracy (simulated)"], nisar_assumed: ["NISAR accuracy assumed"],
  n_fields: ["Fields in the village"],
};
function exValueInput(id, key, val) {
  const opts = key === "site" ? [...$("#site").options].map((o) => [o.value, o.textContent]) : EX_VARY[key][1];
  if (opts) return `<select id="${id}">${opts.map(([v, l]) => `<option value="${v}" ${String(v) === String(val) ? "selected" : ""}>${l}</option>`).join("")}</select>`;
  const [lo, hi] = DEFS.limits[key];
  return `<input id="${id}" type="number" min="${lo}" max="${hi}" step="${key === "n_fields" ? 12 : (hi - lo) / 20}" value="${val}">`;
}
function exDefaults(key) {
  const cur = params();
  const opts = key === "site" ? [...$("#site").options].map((o) => [o.value]) : EX_VARY[key][1];
  if (opts) { const a = String(cur[key]); const b = opts.map((o) => String(o[0])).find((v) => v !== a); return [a, b]; }
  const [lo, hi] = DEFS.limits[key], a = +cur[key];
  return [a, +Math.min(hi, Math.max(lo, a * 1.5 || (lo + hi) / 2)).toFixed(2)];
}
function exPanel() {
  if (EX.built) return;
  EX.built = true;
  $("#expanel").innerHTML = `<p class="note" style="margin-top:0">One app run is one village: one draw of fields, farmers, weather noise and satellite errors.
      An experiment runs setting A and setting B on the same villages and reports means with 95 % intervals, and the paired difference.
      All other settings are the sidebar's.</p>
    <div class="plan-ctl">
      <label>Compare <select id="ex_vary">${Object.entries(EX_VARY).map(([k, v]) => `<option value="${k}">${v[0]}</option>`).join("")}</select></label>
      <label>A <span id="ex_a"></span></label><label>B <span id="ex_b"></span></label>
    </div>
    <div class="plan-ctl">
      <label>Villages <select id="ex_k"><option>3</option><option selected>5</option><option>10</option><option>20</option></select></label>
      <label>Fields per village <select id="ex_n"><option value="36">36</option><option value="60" selected>60</option><option value="120">120</option></select></label>
      <label><input type="checkbox" id="ex_mh"> methane</label>
      <label><input type="checkbox" id="ex_cf"> advice vs practice (slow)</label>
      <label>First seed <input id="ex_seed" type="number" value="1" min="0" max="10000" style="width:70px"></label>
      <button id="ex_run" class="primary">Run experiment</button>
    </div>
    <div id="ex_body"></div>`;
  const sync = () => {
    const k = $("#ex_vary").value, [a, b] = exDefaults(k);
    $("#ex_a").innerHTML = exValueInput("ex_va", k, a); $("#ex_b").innerHTML = exValueInput("ex_vb", k, b);
    $("#ex_n").disabled = k === "n_fields" || (k !== "site" && $("#site").value !== "grid");
  };
  $("#ex_vary").onchange = sync;
  sync();
  $("#ex_run").onclick = exRun;
}
const exVal = (k, v) => (["nisar_update"].includes(k) ? v === "true" : EX_VARY[k][1] || k === "site" ? v : +v);
async function exRun() {
  const k = $("#ex_vary").value, base = { ...params() };
  delete base.seed;
  if (!$("#ex_n").disabled) base.n_fields = +$("#ex_n").value;
  const spec = { base, vary: k, a: exVal(k, $("#ex_va").value), b: exVal(k, $("#ex_vb").value), villages: +$("#ex_k").value,
    seed0: +$("#ex_seed").value, methane: $("#ex_mh").checked, counterfactual: $("#ex_cf").checked };
  try {
    const { id } = await (await post("/api/experiment", spec)).json();
    EX.job = id; EX.res = null;
    $("#ex_run").disabled = true;
    $("#ex_body").innerHTML = `<div class="ex-prog"><div class="ex-bar"><span id="ex_bar"></span></div><span id="ex_ptxt" class="muted">starting…</span></div>`;
    clearInterval(EX.timer);
    EX.timer = setInterval(exPoll, 1000);
  } catch (e) { $("#ex_body").innerHTML = `<p class="note">Could not start: ${esc(e.message)}</p>`; }
}
async function exPoll() {
  let j;
  try { j = await (await fetch(`/api/experiment/${EX.job}`)).json(); } catch { return; }
  if (j.status === "running") {
    $("#ex_bar").style.width = `${(100 * j.done) / j.total}%`;
    const eta = j.done ? Math.round((j.seconds / j.done) * (j.total - j.done)) : null;
    $("#ex_ptxt").textContent = `${j.done} of ${j.total} seasons · ${Math.round(j.seconds)} s${eta !== null ? ` · about ${eta} s left` : ""}`;
    return;
  }
  clearInterval(EX.timer); $("#ex_run").disabled = false;
  if (j.status === "failed") { $("#ex_body").innerHTML = `<p class="note">Failed: ${esc(j.error)}</p>`; return; }
  EX.res = j.result;
  if (!EX.res.summary.some((s_) => s_.key === EX.sel)) EX.sel = EX.res.summary[0].key;
  exBody();
}
function exFmt(v, u) { return v === null || v === undefined ? "–" : u === "" ? v.toFixed(3) : u === "d" ? `${v.toFixed(2)} d` : `${v.toFixed(1)} ${u}`; }
function exBody() {
  const X = EX.res, lab = (v) => { const o = EX_VARY[X.vary][1]; const f = o && o.find((p_) => String(p_[0]) === String(v)); return f ? f[1] : String(v); };
  const groups = { model: "Model", sensors: "Satellites", village: "Village", methane: "Methane", advice: "Advice vs practice" };
  let rows = "", last = null;
  for (const S of X.summary) {
    if (S.group !== last) { rows += `<tr class="ex-g"><td colspan="5">${groups[S.group]}</td></tr>`; last = S.group; }
    const ci = (m, lo, hi) => `${exFmt(m, S.unit)}${lo !== null ? `<span class="muted"> (${exFmt(lo, S.unit)} to ${exFmt(hi, S.unit)})</span>` : ""}`;
    const good = S.better === 0 || S.same || S.diff === null ? "" : (S.diff * S.better > 0 ? "better" : "worse");
    rows += `<tr data-k="${S.key}" class="${S.key === EX.sel ? "on" : ""}"><td>${S.label}</td><td>${ci(S.a, S.a_lo, S.a_hi)}</td><td>${ci(S.b, S.b_lo, S.b_hi)}</td>
      <td>${S.same ? '<span class="muted">same</span>' : `<span class="${S.clear ? good : ""}">${ci(S.diff, S.diff_lo, S.diff_hi)}</span>${S.clear ? " *" : ""}`}</td>
      <td>${S.same ? "–" : S.b_wins === null ? "–" : `${S.b_wins} of ${S.n}`}</td></tr>`;
  }
  $("#ex_body").innerHTML = `<div class="plan-ctl"><b>${EX_VARY[X.vary][0]}: A = ${esc(lab(X.a))}, B = ${esc(lab(X.b))}</b>
      <span class="muted">${X.villages} villages (seeds ${X.seed0}–${X.seed0 + X.villages - 1}), ${X.base.n_fields} fields each${X.base.site !== "grid" ? ` on ${esc(X.base.site)}` : ""} · ${X.seconds} s</span>
      <a class="btnlink" href="/api/experiment/${EX.job}/export.csv" download>Villages (.csv)</a>
      <a class="btnlink" href="/api/experiment/${EX.job}/export.md" download>Summary to cite (.md)</a></div>
    <div class="cf-grid"><div style="grid-column:1/-1"><table class="ex-table"><tr><th>Measure</th><th>A (95 % CI)</th><th>B (95 % CI)</th><th>B − A (95 % CI)</th><th>B better in</th></tr>${rows}</table>
      <p class="note">* the paired 95 % interval excludes zero; blue: B is better, orange: B is worse. Intervals are Student t over villages.
      "Same": the setting does not touch this measure. Click a measure to see each village. Every number is simulation.</p></div>
      <div><h3 id="ex_ch_t"></h3><svg id="ex_chart" viewBox="0 0 420 240" role="img" aria-label="Each village under A and B"></svg></div>
      <div><h3>Settings</h3><p class="note">${Object.entries(X.base).map(([k, v]) => `${k} = ${esc(String(v))}`).join(" · ")}</p></div></div>`;
  document.querySelectorAll(".ex-table tr[data-k]").forEach((tr) => (tr.onclick = () => { EX.sel = tr.dataset.k; exBody(); }));
  exChart();
}
function exChart() {
  const X = EX.res, S = X.summary.find((s_) => s_.key === EX.sel), W = 420, H = 240, m = { l: 56, r: 20, t: 14, b: 30 };
  $("#ex_ch_t").textContent = `${S.label}: each village, A → B`;
  const A = new Map(X.rows.filter((r) => r.arm === "A").map((r) => [r.village, r[S.key]]));
  const B = new Map(X.rows.filter((r) => r.arm === "B").map((r) => [r.village, r[S.key]]));
  const vals = [...A.values(), ...B.values()].filter((v) => v !== null && v !== undefined);
  if (!vals.length) { $("#ex_chart").innerHTML = ""; return; }
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (hi - lo < 1e-9) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * 0.1; lo -= pad; hi += pad;
  const xA = m.l + 60, xB = W - m.r - 60, Y = (v) => m.t + ((H - m.t - m.b) * (hi - v)) / (hi - lo);
  let s = "";
  for (let i = 0; i <= 4; i++) { const v = lo + ((hi - lo) * i) / 4; s += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="${colors.line}"/><text x="${m.l - 6}" y="${Y(v) + 3}" text-anchor="end">${exFmt(v, S.unit)}</text>`; }
  for (const [v, a] of A) {
    const b = B.get(v);
    if (a === null || b === null || a === undefined || b === undefined) continue;
    const c = S.better === 0 || Math.abs(b - a) < 1e-12 ? colors.ink2 : (b - a) * S.better > 0 ? colors.wet : colors.warn;
    s += `<line x1="${xA}" x2="${xB}" y1="${Y(a)}" y2="${Y(b)}" stroke="${c}" stroke-width="1.6"/><circle cx="${xA}" cy="${Y(a)}" r="4" fill="${c}"/><circle cx="${xB}" cy="${Y(b)}" r="4" fill="${c}"/>
      <circle cx="${(xA + xB) / 2}" cy="${(Y(a) + Y(b)) / 2}" r="10" fill="transparent"><title>village ${v}: A ${exFmt(a, S.unit)}, B ${exFmt(b, S.unit)}</title></circle>`;
  }
  s += `<text class="lab" x="${xA}" y="${H - 8}" text-anchor="middle">A</text><text class="lab" x="${xB}" y="${H - 8}" text-anchor="middle">B</text>`;
  $("#ex_chart").innerHTML = s;
}

// ── the farmer's phone (what SmartSoil would show for the selected field) ────
const PH = { tab: "chart", lang: "en", cache: new Map(), answered: new Map(), log: [], seq: 0 };
const BN_DIGITS = "০১২৩৪৫৬৭৮৯";
const bnNum = (s_) => String(s_).replace(/[0-9]/g, (c) => BN_DIGITS[+c]);
// Bangla text is a draft for review by a native speaker before any test with farmers.
const TXT = {
  en: {
    app: "SmartSoil", field: "Field", evening: (d) => `Advice for tomorrow · ${d}`,
    irrigateBy: (d) => `Irrigate by tomorrow, ${d}`, noWater: "No standing water in your field now",
    irrigateMorning: "Irrigate tomorrow morning", noNeed: "No irrigation needed tomorrow",
    lastsUntil: (d) => `Water should last until about ${d}`, range: (a, b) => `Likely between ${a} and ${b}`,
    beyond: "more than two weeks", sure: (p) => `${p} % sure`,
    askQ: "How much water is in your field now?", askWhy: "Your answer makes the advice more certain.",
    cats: ["None", "A little", "Some", "A lot"], catHint: ["dry surface", "up to 2 cm", "2–5 cm", "over 5 cm"],
    thanks: "Thank you. The advice is updated.", before: (t) => `Before your answer: ${t}`,
    tellNow: "Tell us the water now", leakT: "Water loss", leak: (x) => `Your field loses about ${x} mm of water a day`,
    leakHigh: (r) => `${r}× the village average: check the bunds and the puddling`,
    leakLow: (r) => `${r}× the village average: water lasts longer than in most fields`, leakAvg: "About the village average",
    nisar: (w) => `The NISAR satellite saw ${w ? "water" : "no water"} in your field this morning`,
    logT: "Tell SmartSoil", irrigated: "I irrigated today", gone: "The water is gone", pumpH: "Pump hours", save: "Save",
    saved: "Saved (demo: not sent anywhere)", replayNote: "Replay: the farmer's real answers come from the app log",
    sim: "Simulation: what the app would show this evening.", draft: "",
    loading: "Loading…",
  },
  bn: {
    app: "স্মার্টসয়েল", field: "জমি", evening: (d) => `আগামীকালের পরামর্শ · ${d}`,
    irrigateBy: (d) => `আগামীকাল ${d}-এর মধ্যে সেচ দিন`, noWater: "এখন আপনার জমিতে দাঁড়ানো পানি নেই",
    irrigateMorning: "আগামীকাল সকালে সেচ দিন", noNeed: "আগামীকাল সেচ লাগবে না",
    lastsUntil: (d) => `আনুমানিক ${d} পর্যন্ত জমিতে পানি থাকবে`, range: (a, b) => `সম্ভবত ${a} থেকে ${b}-এর মধ্যে`,
    beyond: "দুই সপ্তাহের বেশি", sure: (p) => `${bnNum(p)}% নিশ্চিত`,
    askQ: "এখন আপনার জমিতে কতটা পানি আছে?", askWhy: "আপনার উত্তরে পরামর্শ আরও নিশ্চিত হবে।",
    cats: ["নেই", "অল্প", "মোটামুটি", "অনেক"], catHint: ["মাটি শুকনো", "২ সেমি পর্যন্ত", "২–৫ সেমি", "৫ সেমির বেশি"],
    thanks: "ধন্যবাদ। পরামর্শ হালনাগাদ হয়েছে।", before: (t) => `আপনার উত্তরের আগে: ${t}`,
    tellNow: "এখনকার পানির খবর দিন", leakT: "পানি হারানো", leak: (x) => `আপনার জমি দিনে প্রায় ${bnNum(x)} মিমি পানি হারায়`,
    leakHigh: (r) => `গ্রামের গড়ের ${bnNum(r)} গুণ: আইল ও কাদা করা পরীক্ষা করুন`,
    leakLow: (r) => `গ্রামের গড়ের ${bnNum(r)} গুণ: বেশিরভাগ জমির চেয়ে পানি বেশি দিন থাকে`, leakAvg: "গ্রামের গড়ের কাছাকাছি",
    nisar: (w) => `আজ সকালে NISAR স্যাটেলাইট আপনার জমিতে ${w ? "পানি দেখেছে" : "পানি দেখেনি"}`,
    logT: "স্মার্টসয়েলকে জানান", irrigated: "আজ সেচ দিয়েছি", gone: "পানি শুকিয়ে গেছে", pumpH: "পাম্প চলেছে (ঘণ্টা)", save: "জমা দিন",
    saved: "জমা হয়েছে (ডেমো: কোথাও পাঠানো হয়নি)", replayNote: "রিপ্লে: কৃষকের আসল উত্তর অ্যাপের লগ থেকে আসে",
    sim: "সিমুলেশন: আজ সন্ধ্যায় অ্যাপ যা দেখাত।", draft: "Bangla text: draft, to be checked by a native speaker",
    loading: "লোড হচ্ছে…",
  },
};
const WD = { en: ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"], bn: ["রবিবার", "সোমবার", "মঙ্গলবার", "বুধবার", "বৃহস্পতিবার", "শুক্রবার", "শনিবার"] };
const MON = { en: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
  bn: ["জানুয়ারি", "ফেব্রুয়ারি", "মার্চ", "এপ্রিল", "মে", "জুন", "জুলাই", "আগস্ট", "সেপ্টেম্বর", "অক্টোবর", "নভেম্বর", "ডিসেম্বর"] };
function phDate(k, lang) {   // season day index -> "Thu 16 Feb" / "বৃহস্পতিবার ১৬ ফেব্রুয়ারি"
  const t = new Date(`${R.start_date}T12:00:00`);
  t.setDate(t.getDate() + k);
  return lang === "bn" ? `${WD.bn[t.getDay()]} ${bnNum(t.getDate())} ${MON.bn[t.getMonth()]}` : `${WD.en[t.getDay()]} ${t.getDate()} ${MON.en[t.getMonth()]}`;
}
function phAdvice(s_, d, L, lang) {   // the main message from a belief summary {p_wet, p_dry_tomorrow, dry_80}
  const pct = (p) => Math.max(50, Math.min(99, Math.round(p * 100)));
  if (s_.p_wet < 0.5) return { title: L.noWater, sub: L.irrigateMorning, sure: L.sure(pct(1 - s_.p_wet)), urgent: true };
  if (s_.p_dry_tomorrow !== null && s_.p_dry_tomorrow >= 0.5) return { title: L.irrigateBy(phDate(d + 1, lang)), sub: "", sure: L.sure(pct(s_.p_dry_tomorrow)), urgent: true };
  const day = (D) => (D >= 15 ? L.beyond : phDate(Math.floor(d + 1 + D), lang));
  const q = s_.dry_80;
  return { title: L.noNeed, sub: `${L.lastsUntil(day(q[1]))}<br><span class="ph-small">${L.range(day(q[0]), day(q[2]))}</span>`,
    sure: s_.p_dry_tomorrow === null ? "" : L.sure(pct(1 - s_.p_dry_tomorrow)), urgent: false };
}
async function phoneShow() {
  const box = $("#phone");
  const on = PH.tab === "phone" && sel !== null && R;
  box.hidden = !on;
  $("#chart").style.display = on ? "none" : "";
  if (!on) return;
  const f = sel, d = day, L = TXT[PH.lang], key = `${R.run}|${f}|${d}`, F = R.fields[f], seqNo = ++PH.seq;
  if (!planted(f, d)) { phoneRender({ f, d, msg: null }); return; }
  let s_ = null, after = null;
  if (replay()) {
    const pw = R.belief.p_wet[f][d], dp = R.dry_pred[f][d];
    s_ = { p_wet: pw ?? 0, p_dry_tomorrow: null, dry_80: [dp ?? 0, dp ?? 0, dp ?? 0] };
  } else {
    if (!PH.cache.has(key)) {
      phoneRender({ f, d, loading: true });
      try { PH.cache.set(key, await (await post("/api/ask", { run: R.run, field: f, day: d, kind: "none" })).json()); } catch (e) { return; }
      if (seqNo !== PH.seq) return;
    }
    s_ = PH.cache.get(key).before;
    after = PH.answered.get(key) || null;
  }
  phoneRender({ f, d, s: s_, after });
}
function phoneRender({ f, d, s: s_, after, loading }) {
  const L = TXT[PH.lang], lang = PH.lang, F = R.fields[f];
  let body;
  if (loading) body = `<div class="ph-card">${L.loading}</div>`;
  else if (!s_) body = `<div class="ph-card ph-muted">${lang === "bn" ? "এখনও রোপণ হয়নি" : "Not transplanted yet"}</div>`;
  else {
    const now = after ? after.after : s_, m = phAdvice(now, d, L, lang);
    const uncertain = !after && !replay() && ((s_.p_dry_tomorrow > 0.2 && s_.p_dry_tomorrow < 0.8) || (s_.p_wet > 0.2 && s_.p_wet < 0.8));
    const med = R.fields.map((x) => x.sp_model).sort((a, b) => a - b)[Math.floor(R.n / 2)], ratio = F.sp_model / med;
    const rTxt = (Math.round(ratio * 10) / 10).toFixed(1), x = (Math.round(F.sp_model * 10) / 10).toFixed(1);
    const nis = nisarNext(f, d);
    body = `<div class="ph-card ph-main ${m.urgent ? "urgent" : ""}"><div class="ph-title">${m.title}</div>${m.sub ? `<div class="ph-sub">${m.sub}</div>` : ""}
        ${m.sure ? `<div class="ph-sure">${m.sure}</div>` : ""}
        ${after ? `<div class="ph-thanks">${L.thanks}<br><span class="ph-small">${L.before(phAdvice(s_, d, L, lang).title)}</span></div>` : ""}</div>
      ${nis ? `<div class="ph-card ph-sat">🛰 ${L.nisar(nis.wet)}</div>` : ""}
      ${replay() ? `<div class="ph-card ph-muted">${L.replayNote}</div>` : uncertain ? phAskCard(L) :
        after ? "" : `<button class="ph-link" id="ph_tell">${L.tellNow}</button><div id="ph_askbox"></div>`}
      <div class="ph-card"><div class="ph-h">${L.leakT}</div>${L.leak(x)}<div class="ph-small">${ratio >= 1.5 ? L.leakHigh(rTxt) : ratio <= 0.67 ? L.leakLow(rTxt) : L.leakAvg}</div></div>
      <div class="ph-card"><div class="ph-h">${L.logT}</div><div class="ph-btns"><button id="ph_irr">${L.irrigated}</button><button id="ph_gone">${L.gone}</button></div><div id="ph_logbox"></div></div>`;
  }
  $("#phone").innerHTML = `<div class="ph-wrap"><div class="ph-lang seg"><button data-l="en" class="${lang === "en" ? "on" : ""}">English</button><button data-l="bn" class="${lang === "bn" ? "on" : ""}">বাংলা</button></div>
    <div class="ph-frame" lang="${lang === "bn" ? "bn" : "en"}"><div class="ph-notch"></div><div class="ph-status"><span>${lang === "bn" ? bnNum("18:30") : "18:30"}</span><span>▮▮▮ 4G</span></div>
      <div class="ph-app"><b>${L.app}</b><span>${L.field} ${lang === "bn" ? bnNum(F.id) : F.id}</span></div>
      <div class="ph-date">${L.evening(phDate(d + 1, lang))}</div>${body}
      <div class="ph-foot">${L.sim}${L.draft ? `<br>${L.draft}` : ""}</div></div></div>`;
  document.querySelectorAll(".ph-lang button").forEach((b) => (b.onclick = () => { PH.lang = b.dataset.l; phoneShow(); }));
  const wireAsk = () => document.querySelectorAll(".ph-cat").forEach((b) => (b.onclick = () => phAnswer(f, d, +b.dataset.c)));
  wireAsk();
  if ($("#ph_tell")) $("#ph_tell").onclick = () => { $("#ph_askbox").innerHTML = phAskCard(L); $("#ph_tell").remove(); wireAsk(); };
  if ($("#ph_irr")) $("#ph_irr").onclick = () => {
    $("#ph_logbox").innerHTML = `<label class="ph-small">${L.pumpH} <input id="ph_hours" type="number" min="0" max="24" step="0.5" value="2"></label> <button id="ph_save">${L.save}</button>`;
    $("#ph_save").onclick = () => { PH.log.push({ field: F.id, day: d, event: "irrigated", hours: +$("#ph_hours").value }); $("#ph_logbox").innerHTML = `<div class="ph-small">${L.saved}</div>`; };
  };
  if ($("#ph_gone")) $("#ph_gone").onclick = () => { PH.log.push({ field: F.id, day: d, event: "water_gone" }); $("#ph_logbox").innerHTML = `<div class="ph-small">${L.saved}</div>`; };
}
function phAskCard(L) {
  return `<div class="ph-card ph-ask"><div class="ph-h">${L.askQ}</div><div class="ph-small">${L.askWhy}</div>
    <div class="ph-cats">${L.cats.map((c, i) => `<button class="ph-cat" data-c="${i}"><b>${c}</b><span>${L.catHint[i]}</span></button>`).join("")}</div></div>`;
}
async function phAnswer(f, d, cat) {
  try {
    const a = await (await post("/api/ask", { run: R.run, field: f, day: d, kind: "depth", answer: cat })).json();
    PH.answered.set(`${R.run}|${f}|${d}`, a);
    asks.set(f, [...(asks.get(f) || []).filter((x) => x.day !== a.day), a]);
    phoneShow();
  } catch (e) { $("#phone").insertAdjacentHTML("beforeend", `<p class="note">${esc(e.message)}</p>`); }
}

// ── the crop on the selected field today, and what it does to the satellites (Experiment 27) ──
function cropLine() {
  if (!R || sel === null || !$("#fcrop")) return;
  const f = sel, cp = R.crop, T = R.sensing_tables, lai = R.lai[f][day];
  if (!cp || cp.stage[f][day] < 0) { $("#fcrop").textContent = "not transplanted yet"; $("#fsens").textContent = "–"; return; }
  const dat = Math.max(0, Math.floor(day + 0.5 - R.fields[f].tp));
  $("#fcrop").textContent = `day ${dat} after transplanting · ${cp.stages[cp.stage[f][day]]} · LAI ${lai.toFixed(1)} · ` +
    `${Math.round(cp.height_cm[f][day])} cm tall · ${Math.round(cp.tillers[f][day])} tillers per hill · canopy covers ${Math.round(cp.cover[f][day] * 100)} %`;
  if (!T) { $("#fsens").textContent = "–"; return; }
  const bin = (lo, x) => { let k = 0; lo.forEach((v, i) => { if (x >= v) k = i; }); return k; };
  // as the simulator detects (paddy_sim._detect): no worse than a coin toss under a closed canopy
  const s2 = Math.max(0.5, T.s2_recall[bin(T.s2_lai_lo, lai)]), s1 = Math.max(0.5, T.s1_acc[bin(T.s1_lai_lo, lai)]);
  const h = cp.height_cm[f][day], ni = Math.max(0.55, T.nisar_acc - T.nisar_drop * Math.max(0, (h - T.nisar_full_cm) / T.nisar_drop_cm));
  $("#fsens").innerHTML = `under this canopy, standing water is seen by Sentinel-2 ${Math.round(s2 * 100)} % of the time${s2 <= 0.5 ? " (no better than a coin toss)" : ""} · ` +
    `Sentinel-1 calls right ${Math.round(s1 * 100)} % · NISAR ${Math.round(ni * 100)} %${h > T.nisar_full_cm ? " (plants above 70 cm)" : ""}`;
}

// ── loading card: real progress from the server (/api/progress), then download and drawing ──
const LD_SEASON = ["Laying out the village and simulating the season", "The model's belief, day by day", "Forecasting when each field runs dry",
  "True forecasts, satellite checks and the crop", "Packing the results", "Receiving the results", "Drawing the village"];
const LD_REPLAY = ["Reading the campaign files", "The model's belief, day by day", "Forecasting when each field runs dry",
  "Logger truth, satellite checks and data QC", "Packing the results", "Receiving the results", "Drawing the village"];
const LD_SERVER = 0.86;   // the share of the bar for the server's work; the rest is download and drawing
let LD = null;
function loaderStart(title, steps) {
  if (LD) { clearInterval(LD.clock); LD.stopPolling(); }   // a newer run replaces the card's timers
  const el = $("#loader");
  el.hidden = false; el.classList.remove("fade");
  document.querySelector(".mapwrap").classList.add("loading");
  $("#ld_title").textContent = title;
  $("#ld_err").hidden = true;
  $("#ld_steps").innerHTML = steps.map((s_) => `<li>${s_}</li>`).join("");
  const ld = { token: Math.random().toString(36).slice(2, 12), steps, t0: performance.now(), frac: 0, waiting: false, timer: null };
  ld.clock = setInterval(() => { $("#ld_time").textContent = `${((performance.now() - ld.t0) / 1000).toFixed(1)} s${ld.waiting ? " · waiting for the previous run" : ""}`; }, 200);
  ld.stopPolling = () => clearInterval(ld.timer);
  ld.timer = setInterval(async () => {
    try {
      const p = await (await fetch(`/api/progress/${ld.token}`)).json();
      ld.waiting = !!p.waiting;
      if (p.stages) loaderStep(ld, p.stage, p.frac * LD_SERVER);
    } catch { /* the next poll will tell */ }
  }, 300);
  loaderStep(ld, 0, 0.01);
  LD = ld;
  return ld;
}
function loaderStep(ld, i, frac) {
  ld.frac = Math.max(ld.frac, frac);
  $("#ld_bar").style.width = `${(ld.frac * 100).toFixed(1)}%`;
  $("#ld_pct").textContent = `${Math.round(ld.frac * 100)} %`;
  document.querySelectorAll("#ld_steps li").forEach((li, k) => { li.className = k < i ? "ok" : k === i ? "now" : ""; });
}
async function loaderReceive(ld, res) {   // stream the response so the bar moves while a large season downloads
  const i = ld.steps.length - 2, total = +res.headers.get("content-length") || 0;
  loaderStep(ld, i, LD_SERVER);
  if (!res.body || !res.body.getReader) return res.json();
  const reader = res.body.getReader(), chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length;
    const mb = (got / 1e6).toFixed(1);
    $("#ld_steps li:nth-child(" + (i + 1) + ")").textContent = `Receiving the results (${mb}${total ? ` of ${(total / 1e6).toFixed(1)}` : ""} MB)`;
    loaderStep(ld, i, LD_SERVER + (total ? (got / total) * 0.08 : 0.04));
  }
  const buf = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { buf.set(c, o); o += c.length; }
  return JSON.parse(new TextDecoder().decode(buf));
}
function loaderEnd(ld) {
  clearInterval(ld.clock); ld.stopPolling();
  loaderStep(ld, ld.steps.length, 1);
  const el = $("#loader");
  el.classList.add("fade");
  setTimeout(() => { if (el.classList.contains("fade")) { el.hidden = true; document.querySelector(".mapwrap").classList.remove("loading"); } }, 260);
}
function loaderFail(ld, msg) {
  clearInterval(ld.clock);
  $("#ld_err").hidden = false;
  $("#ld_err").textContent = `Failed: ${msg.slice(0, 300)}. Change a setting or press Run season to try again.`;
}

init();
