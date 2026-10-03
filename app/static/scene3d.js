// Paddy Twin 3D: the simulated village as a 3D scene (three.js).
//
// Everything shown is driven by the season from /api/season:
//   water in each field  = the simulator's water level (vertical scale exaggerated, see VY)
//   soil colour          = wet mud when ponded, drying and cracking as the level falls below the surface
//   rice                 = height, tillers (hill width), colour and grain heads from the crop model (Experiment 27)
//   tubewells, channels  = run on days a field in that command area is irrigated; the inlet to the field opens
//   rain                 = ERA5 rain of the day
//   satellites           = NISAR (06:00, L-band SAR), Sentinel-2 (10:30, optical), Sentinel-1 (18:00, C-band SAR)
//                          cross on their acquisition days, in that order, under the matching sun; each field
//                          they see gets a ring in the sensor's colour (red if the wet/dry call was wrong)
//   overlays             = model belief (orange sheet at the model's water level), alerts, satellite passes
//   loggers              = water-level loggers (AWD pipe + logger head) at the site's positions, with their reading
// The layout comes from the run (R.layout, local metres): the generated grid or a site file in app/sites/ with
// surveyed field polygons, logger positions, tubewells and channels (see app/sites.py).

import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";
import { Sky } from "three/addons/Sky.js";

const U = 1 / 3;        // scene units per metre (1 unit = 3 m)
const BUND = 0.8;       // bund width (2.4 m drawn)
const VY = 0.004;       // scene units per mm of water (1 mm drawn as 12 mm: water depth exaggerated ×12)
const BUND_H = 0.75;
const RICE_H = 1.5;     // rice height at full canopy (4.5 m drawn for about 1 m: ×4.5)
// satellites in time-of-day order; frame inset (scene units) separates their rings on the same field
const SATS = {
  nisar: { label: "NISAR", time: "06:00", band: "L-band SAR", color: "#8b5cf6", inset: 1.15, desc: true, sun: [5, 82, "#ffb489", 0.45] },
  s2: { label: "Sentinel-2", time: "10:30", band: "optical", color: "#1b9fc6", inset: 0.1, desc: true, sun: [48, 125, "#fff4e0", 1.0] },
  s1: { label: "Sentinel-1", time: "18:00", band: "C-band SAR", color: "#e07b2c", inset: 0.62, desc: false, sun: [5, 278, "#ff9d5c", 0.4] },
};
const SAT_ORDER = ["nisar", "s2", "s1"];
const isWrong = (p) => p.truth !== null && p.truth !== undefined && p.wet !== p.truth;
const SEASON_COLORS = {
  wetMud: new THREE.Color("#3f3226"), moist: new THREE.Color("#5c4733"), dry: new THREE.Color("#b79e78"),
  bare: new THREE.Color("#a58d6b"), shallow: new THREE.Color("#7b8763"), deep: new THREE.Color("#3d6d8c"),
  young: new THREE.Color("#93c95a"), green: new THREE.Color("#3b8a2c"), ripe: new THREE.Color("#cdb24e"),
  panicleGreen: new THREE.Color("#a7b84a"), panicleRipe: new THREE.Color("#d9b347"),
};

export class Village3D {
  constructor(container, { onSelect } = {}) {
    this.container = container;
    this.onSelect = onSelect;
    this.visible = false;
    this.show = { signals: true, belief: false, loggers: true, nisar: true, s2: true, s1: true, imagery: true };
    this.imgTex = new Map();   // image url -> texture (real Sentinel-2 ground, loaded on first use)
    this.uTime = { value: 0 };
    this.clock = new THREE.Clock();

    const r = (this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" }));
    r.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    r.toneMapping = THREE.ACESFilmicToneMapping;
    r.toneMappingExposure = 0.55;
    r.shadowMap.enabled = true;
    r.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(r.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.5, 6000);
    this.controls = new OrbitControls(this.camera, r.domElement);
    this.controls.enableDamping = true;
    this.controls.maxPolarAngle = 1.48;
    this.controls.minDistance = 6;
    this.controls.maxDistance = 900;

    this.tex = makeTextures();
    this.initSky();
    this.initHud();
    this.initPointer();
    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
    this.loop = this.loop.bind(this);
    requestAnimationFrame(this.loop);
  }

  // ── environment ────────────────────────────────────────────────────────
  initSky() {
    const sky = (this.sky = new Sky());
    sky.scale.setScalar(5000);
    const u = sky.material.uniforms;
    u.turbidity.value = 7; u.rayleigh.value = 1.6; u.mieCoefficient.value = 0.006; u.mieDirectionalG.value = 0.82;
    this.scene.add(sky);
    this.sunDir = new THREE.Vector3().setFromSphericalCoords(1, THREE.MathUtils.degToRad(90 - 36), THREE.MathUtils.degToRad(150));
    u.sunPosition.value.copy(this.sunDir);

    this.hemi = new THREE.HemisphereLight("#cfe3ff", "#5a4a32", 0.9);
    this.scene.add(this.hemi);
    const sun = (this.sun = new THREE.DirectionalLight("#fff1d6", 3.2));
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.bias = -0.0004;
    sun.shadow.normalBias = 0.04;
    this.scene.add(sun, sun.target);
    this.scene.fog = new THREE.Fog("#c9d6e3", 300, 1400);

    // reflections in the water come from the sky
    const pm = new THREE.PMREMGenerator(this.renderer);
    const envScene = new THREE.Scene();
    const s2 = new Sky(); s2.scale.setScalar(5000);
    Object.assign(s2.material.uniforms.turbidity, { value: 7 });
    s2.material.uniforms.sunPosition.value.copy(this.sunDir);
    envScene.add(s2);
    this.scene.environment = pm.fromScene(envScene, 0.04).texture;
    pm.dispose();
  }

  initHud() {
    const h = (this.hud = document.createElement("div"));
    h.className = "hud3d";
    h.innerHTML = `
      <div class="hud-tl"><div class="hud-date"></div><div class="hud-wx"></div><div class="hud-sat"></div><div class="hud-site"></div></div>
      <div class="hud-tr">
        <label><input type="checkbox" data-k="signals" checked> Satellite passes &amp; alerts</label>
        <div class="hud-sensors">${SAT_ORDER.map((k) => `<label style="--c:${SATS[k].color}"><input type="checkbox" data-k="${k}" checked> ${SATS[k].label}</label>`).join("")}</div>
        <label><input type="checkbox" data-k="belief"> Model belief (orange outline at the model's water level)</label>
        <label><input type="checkbox" data-k="loggers" checked> Water-level loggers</label>
        <label class="hud-img"><input type="checkbox" data-k="imagery" checked> Real Sentinel-2 image around the village</label>
        <button type="button" data-a="reset">Reset view</button>
      </div>
      <div class="hud-bl">Heights exaggerated (water ×12, bunds ×7, rice ×4.5) so a few centimetres of water show · drag to orbit,
        right-drag to pan, scroll to zoom, click a field, double-click to fly to it</div>`;
    this.container.appendChild(h);
    h.querySelectorAll("input[data-k]").forEach((el) => (el.onchange = () => { this.show[el.dataset.k] = el.checked; this.applyDay(); }));
    h.querySelector("[data-a=reset]").onclick = () => this.resetView();
  }

  initPointer() {
    const el = this.renderer.domElement;
    const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
    let down = null;
    const pick = (e) => {
      if (!this.v) return null;
      const rect = el.getBoundingClientRect();
      ndc.set(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
      ray.setFromCamera(ndc, this.camera);
      const v = this.v;
      const hit = ray.intersectObjects([v.water, v.soil, v.rice, ...v.loggerPick], false)[0];
      if (!hit) return null;
      if (hit.object === v.rice) return hit.instanceId === undefined ? null : v.hillField[hit.instanceId];
      if (hit.object.userData.field !== undefined) return hit.object.userData.field;
      return hit.faceIndex === undefined ? null : v.faceField[hit.faceIndex];
    };
    el.addEventListener("pointerdown", (e) => (down = [e.clientX, e.clientY]));
    el.addEventListener("pointerup", (e) => {
      if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) return;
      const f = pick(e);
      if (f !== null && f !== undefined && this.onSelect) this.onSelect(f);
    });
    el.addEventListener("dblclick", (e) => { const f = pick(e); if (f !== null && f !== undefined) this.flyTo(f); });
  }

  // ── village geometry, from the run's layout (metres) ───────────────────
  setData(R) {
    this.R = R;
    if (this.v) { this.scene.remove(this.v.group); disposeGroup(this.v.group); }
    const v = (this.v = { group: new THREE.Group() });
    this.scene.add(v.group);
    const L = R.layout, n = R.n;
    const rng = mulberry((R.params.seed || 1) * 7919 + n);
    v.fields = L.fields.map((fl) => {
      const pts = fl.poly.map(([x, z]) => [x * U, z * U]);
      return { pts, c: [fl.c[0] * U, fl.c[1] * U], tw: fl.tw };
    });
    let x0 = Infinity, x1 = -Infinity, z0 = Infinity, z1 = -Infinity;
    v.fields.forEach((fl) => fl.pts.forEach(([x, z]) => { x0 = Math.min(x0, x); x1 = Math.max(x1, x); z0 = Math.min(z0, z); z1 = Math.max(z1, z); }));
    v.bbox = { x0, x1, z0, z1 };
    v.W = x1 - x0; v.D = z1 - z0;
    v.center = new THREE.Vector3((x0 + x1) / 2, 0, (z0 + z1) / 2);
    const cx = v.center.x, cz = v.center.z;

    // ground and the village earth platform
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(4000, 4000), new THREE.MeshStandardMaterial({ map: this.tex.grass, roughness: 1 }));
    ground.rotation.x = -Math.PI / 2; ground.position.set(cx, -0.32, cz); ground.receiveShadow = true;
    const earth = new THREE.Mesh(new THREE.BoxGeometry(v.W + 40, 0.3, v.D + 30), new THREE.MeshStandardMaterial({ map: this.tex.dirt, roughness: 1 }));
    earth.position.set(cx - 5, -0.3, cz); earth.receiveShadow = true;
    v.group.add(ground, earth);

    // per-field surfaces: soil (colour per day), water and the model's belief (height per frame), pass frames
    const tris = v.fields.map((fl) => THREE.ShapeUtils.triangulateShape(fl.pts.map(([x, z]) => new THREE.Vector2(x, z)), []));
    v.faceField = new Int32Array(tris.reduce((s, t) => s + t.length, 0));
    tris.reduce((k, t, f) => { v.faceField.fill(f, k, k + t.length); return k + t.length; }, 0);
    const soilG = polyGeometry(v.fields, tris, 0, true);
    v.ranges = soilG.userData.ranges;
    v.soil = new THREE.Mesh(soilG, new THREE.MeshStandardMaterial({ map: this.tex.soil, vertexColors: true, roughness: 0.95 }));
    v.water = new THREE.Mesh(polyGeometry(v.fields, tris, -0.5, true), new THREE.MeshStandardMaterial({
      vertexColors: true, transparent: true, opacity: 0.9, roughness: 0.72, metalness: 0.0,
      normalMap: this.tex.ripple, normalScale: new THREE.Vector2(0.06, 0.06), envMapIntensity: 0.18 }));
    v.belief = new THREE.Mesh(polyGeometry(v.fields, tris, -0.5, false), new THREE.MeshBasicMaterial({
      color: "#ff7a2e", transparent: true, opacity: 0.22, depthWrite: false, side: THREE.DoubleSide }));
    const inset = v.fields.map((fl) => insetPoly(fl.pts, BUND / 2 + 0.2));
    v.beliefRim = new THREE.Mesh(ribbonGeometry(inset, 0.32, -0.5, true), new THREE.MeshBasicMaterial({ color: "#ff6a1a", side: THREE.DoubleSide }));
    v.frames = {};
    for (const k of SAT_ORDER) {
      v.frames[k] = new THREE.Mesh(ribbonGeometry(v.fields.map((fl) => insetPoly(fl.pts, SATS[k].inset)), 0.32, BUND_H + 0.06, true, true),
        new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false, side: THREE.DoubleSide }));
      v.group.add(v.frames[k]);
    }
    v.soil.receiveShadow = v.water.receiveShadow = true;
    v.group.add(v.soil, v.water, v.belief, v.beliefRim);
    v.waterH = new Float32Array(n).fill(-1);
    v.beliefH = new Float32Array(n).fill(-1);
    v.fresh = true;

    // bunds along every edge (shared edges overlap and read as one bund)
    const edges = [];
    v.fields.forEach((fl) => fl.pts.forEach((a, i) => edges.push([a, fl.pts[(i + 1) % fl.pts.length]])));
    v.bunds = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshStandardMaterial({ map: this.tex.bund, roughness: 1 }), edges.length);
    const dummy = new THREE.Object3D();
    edges.forEach(([a, b], k) => {
      const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
      dummy.position.set((a[0] + b[0]) / 2, BUND_H / 2 - 0.3, (a[1] + b[1]) / 2);
      dummy.rotation.set(0, -Math.atan2(b[1] - a[1], b[0] - a[0]), 0);
      dummy.scale.set(len + BUND * 0.6, BUND_H + 0.3, BUND); dummy.updateMatrix();
      v.bunds.setMatrixAt(k, dummy.matrix);
    });
    v.bunds.castShadow = v.bunds.receiveShadow = true;
    v.group.add(v.bunds);

    // rice hills in rows along each plot's longest side
    const riceMat = new THREE.MeshStandardMaterial({ vertexColors: true, side: THREE.DoubleSide, roughness: 0.75 });
    riceMat.onBeforeCompile = (sh) => {
      sh.uniforms.uTime = this.uTime;
      sh.vertexShader = "uniform float uTime;\n" + sh.vertexShader.replace("#include <begin_vertex>", `#include <begin_vertex>
        float ph = instanceMatrix[3].x * 0.37 + instanceMatrix[3].z * 0.23;
        float sway = sin(uTime * 1.4 + ph) * 0.07 + sin(uTime * 2.7 + ph * 1.9) * 0.03;
        transformed.x += sway * position.y * position.y;
        transformed.z += sway * 0.5 * position.y * position.y;`);
    };
    v.riceBase = []; v.hillRange = [];
    const hillField = [];
    v.fields.forEach((fl, f) => {
      const start = v.riceBase.length;
      for (const [x, z] of hillPositions(fl.pts, rng)) { v.riceBase.push([x, z, rng() * Math.PI * 2, 0.85 + rng() * 0.3]); hillField.push(f); }
      v.hillRange.push([start, v.riceBase.length - start]);
    });
    v.hillField = Int32Array.from(hillField);
    v.rice = new THREE.InstancedMesh(riceHillGeometry(), riceMat, Math.max(1, v.riceBase.length));
    v.rice.receiveShadow = true;
    // grain heads (panicles): hidden until heading, then green, golden as the grain fills
    v.panicles = new THREE.InstancedMesh(panicleGeometry(), riceMat, Math.max(1, v.riceBase.length));
    v.group.add(v.rice, v.panicles);

    // tubewells, channels, field inlets, roads
    v.channels = [];
    for (const ch of L.channels) {
      const pts = ch.pts.map(([x, z]) => [x * U, z * U]), w = ch.width * U;
      const bed = new THREE.Mesh(ribbonGeometry([pts], w * 0.8, -0.12, false), new THREE.MeshStandardMaterial({ color: "#4d3b2a", roughness: 1 }));
      const water = new THREE.Mesh(ribbonGeometry([pts], w * 0.62, 0.12, false), new THREE.MeshStandardMaterial({
        map: this.tex.flow, color: "#9fc3d6", transparent: true, opacity: 0.85, roughness: 0.1, envMapIntensity: 1.2 }));
      v.group.add(bed, water);
      for (const side of [-1, 1]) {
        const emb = new THREE.Mesh(ribbonGeometry([offsetLine(pts, side * (w / 2 - 0.2))], 0.5, 0.2, false), new THREE.MeshStandardMaterial({ map: this.tex.bund, roughness: 1 }));
        emb.receiveShadow = true;
        v.group.add(emb);
      }
      v.channels.push({ water, tw: ch.tw, pts });
    }
    for (const road of L.roads) {
      const m = new THREE.Mesh(ribbonGeometry([road.pts.map(([x, z]) => [x * U, z * U])], road.width * U, 0.02, false),
        new THREE.MeshStandardMaterial({ map: this.tex.dirt, color: "#d9c7a6", roughness: 1 }));
      m.receiveShadow = true;
      v.group.add(m);
    }
    v.pumps = L.tubewells.map((t, i) => {
      const x = t.x * U, z = t.z * U;
      const ch = v.channels.find((c) => c.tw === i);
      const mine = v.fields.filter((fl) => fl.tw === i);
      const [tx, tz] = ch ? ch.pts[0] : mine.length ? [mine.reduce((s, fl) => s + fl.c[0], 0) / mine.length, mine.reduce((s, fl) => s + fl.c[1], 0) / mine.length] : [x + 1, z];
      const pump = pumpHouse(this.tex);
      pump.position.set(x, 0, z);
      pump.rotation.y = -Math.atan2(tz - z, tx - x);
      const spout = spoutParticles();
      spout.position.set(1.9, 0.9, 0);
      pump.add(spout);
      const sign = textSprite(t.name, "");
      sign.position.set(x, 4.6, z);
      v.group.add(pump, sign);
      return { spout, sign, name: t.name, running: false };
    });
    v.inlets = new THREE.InstancedMesh(new THREE.BoxGeometry(1.1, 0.22, BUND + 0.5), new THREE.MeshStandardMaterial({
      color: "#bfe0f0", transparent: true, opacity: 0.9, roughness: 0.05, emissive: "#355f78", emissiveIntensity: 0.3 }), n);
    v.inletMatrix = v.fields.map((fl) => {
      const ch = v.channels.find((c) => c.tw === fl.tw);
      const pump = L.tubewells[fl.tw];
      const target = ch ? ch.pts : [[pump.x * U, pump.z * U]];
      const { p, ang } = nearestEdgePoint(fl.pts, target);
      dummy.position.set(p[0], 0.32, p[1]); dummy.rotation.set(0, -ang, 0); dummy.scale.set(1, 1, 1); dummy.updateMatrix();
      return dummy.matrix.clone();
    });
    v.group.add(v.inlets);

    // water-level loggers: AWD pipe, logger head with solar panel and antenna, and a reading tag
    v.loggers = []; v.loggerPick = [];
    const lgGeo = loggerGeometries(), lgMat = loggerMaterials();
    for (const lg of R.loggers || []) {
      const g = new THREE.Group();
      g.position.set(lg.x * U, 0, lg.z * U);
      const pipe = new THREE.Mesh(lgGeo.pipe, lgMat.pipe); pipe.position.y = 0.35;
      const head = new THREE.Mesh(lgGeo.head, lgMat.head); head.position.y = 1.45;
      const panel = new THREE.Mesh(lgGeo.panel, lgMat.panel); panel.position.set(0, 1.68, 0); panel.rotation.x = -0.5;
      const ant = new THREE.Mesh(lgGeo.antenna, lgMat.head); ant.position.set(0.15, 2.05, -0.12);
      const led = new THREE.Mesh(lgGeo.led, lgMat.led.clone()); led.position.set(0.22, 1.5, 0.16);
      for (const m of [pipe, head, panel, ant]) { m.castShadow = true; m.userData.field = lg.field; }
      g.add(pipe, head, panel, ant, led);
      const tag = textSprite(`Logger ${lg.id}`, "");
      tag.scale.set(4.4, 1.1, 1); tag.position.y = 2.75;
      g.add(tag);
      v.group.add(g);
      v.loggers.push({ g, tag, led, lg });
      v.loggerPick.push(pipe, head);
    }

    // satellite passes, selection, alert and doubt markers
    v.sel = null;
    v.marks = v.fields.map((fl) => {
      const alert = new THREE.Sprite(this.tex.alertHit); alert.scale.set(2.2, 2.2, 1);
      alert.position.set(fl.c[0] - 1.4, RICE_H + 3.4, fl.c[1]);
      const doubt = new THREE.Sprite(this.tex.doubt); doubt.scale.set(2.2, 2.2, 1);
      doubt.position.set(fl.c[0] + 1.4, RICE_H + 3.4, fl.c[1]);
      alert.visible = doubt.visible = false;
      v.group.add(alert, doubt);
      return { alert, doubt };
    });
    v.passByDay = new Map();   // day -> sensor -> field -> pass
    for (const p of R.passes) {
      if (!v.passByDay.has(p.d)) v.passByDay.set(p.d, new Map());
      const day = v.passByDay.get(p.d);
      if (!day.has(p.s)) day.set(p.s, new Map());
      day.get(p.s).set(p.f, p);
    }

    v.surr = new THREE.Group();
    surroundingsAround(v.surr, rng, v);
    v.group.add(v.surr);
    // real Sentinel-2 true colour (north up, the layout's metres) under and around the village, if the site has it
    v.img = null;
    const I = L.imagery;
    this.hud.querySelector(".hud-img").style.display = I ? "" : "none";
    if (I) {
      const size = 2 * I.half_size_m * U;
      v.img = new THREE.Mesh(new THREE.PlaneGeometry(size, size), new THREE.MeshLambertMaterial({ color: "#ffffff" }));
      v.img.rotation.x = -Math.PI / 2;       // texture top = north = -z
      v.img.position.set(I.center_xz[0] * U, -0.27, I.center_xz[1] * U);
      v.img.receiveShadow = true;
      v.img.userData.dates = I.dates.map((s_) => [new Date(`${s_}T12:00:00`), s_]);
      v.group.add(v.img);
    }
    v.rain = rainLines(v.W + 60, v.D + 60, cx, cz);
    v.sats = {};
    for (const k of SAT_ORDER) {
      const model = satelliteModel(k), swath = new THREE.Mesh(flatPlane(v.W * 1.6, 5), new THREE.MeshBasicMaterial({
        color: SATS[k].color, transparent: true, opacity: 0.25, depthWrite: false }));
      const beamGeo = new THREE.BufferGeometry();
      beamGeo.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(9), 3));
      const beam = new THREE.Mesh(beamGeo, new THREE.MeshBasicMaterial({ color: SATS[k].color, transparent: true, opacity: 0.16,
        depthWrite: false, side: THREE.DoubleSide }));
      beam.frustumCulled = false;
      model.visible = swath.visible = beam.visible = false;
      v.sats[k] = { model, swath, beam };
      v.group.add(model, swath, beam);
    }
    v.group.add(v.rain);

    const half = Math.max(v.W, v.D) * 0.75 + 30;
    Object.assign(this.sun.shadow.camera, { left: -half, right: half, top: half, bottom: -half, near: 1, far: 3 * half + 400 });
    this.sun.shadow.camera.updateProjectionMatrix();
    this.sun.position.copy(v.center).addScaledVector(this.sunDir, half + 200);
    this.sun.target.position.copy(v.center);
    this.scene.fog.near = Math.max(v.W, v.D) * 1.5; this.scene.fog.far = Math.max(v.W, v.D) * 6 + 600;
    if (this.layoutName !== L.name) { this.resetView(); this.layoutName = L.name; }
    this.day = null;
    if (this.selected !== undefined && this.selected !== null && this.selected < n) this.setSelected(this.selected);
  }

  resetView() {
    if (!this.v) return;
    const { center, W, D } = this.v;
    const d = Math.max(W, D);
    this.camera.position.set(center.x - d * 0.42, d * 0.3, center.z + d * 0.7);
    this.controls.target.set(center.x - d * 0.12, 0, center.z + d * 0.18);
    this.flight = null;
  }

  flyTo(f) {
    const [x, z] = this.v.fields[f].c;
    const target = new THREE.Vector3(x, 0, z);
    this.flight = { t: 0, fromPos: this.camera.position.clone(), toPos: target.clone().add(new THREE.Vector3(-14, 12, 22)), fromT: this.controls.target.clone(), toT: target };
  }

  // ── a day ─────────────────────────────────────────────────────────────
  // the latest real image taken on or before the shown day (the first one before it)
  applyImagery() {
    const v = this.v, I = this.R.layout.imagery;
    if (!v.img) { this.imgNote = ""; return; }
    v.img.visible = this.show.imagery;
    v.surr.visible = !this.show.imagery;
    if (!this.show.imagery) { this.imgNote = ""; return; }
    const now = new Date(`${this.R.start_date}T12:00:00`);
    now.setDate(now.getDate() + this.day);
    const ds = v.img.userData.dates, pick = [...ds].reverse().find(([t]) => t <= now) || ds[0];
    const url = `${I.base}${pick[1]}.jpg`;
    if (!this.imgTex.has(url)) {
      const tex = new THREE.TextureLoader().load(url, () => this.requestRender && this.requestRender());
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.anisotropy = 4;
      this.imgTex.set(url, tex);
    }
    if (v.img.material.map !== this.imgTex.get(url)) { v.img.material.map = this.imgTex.get(url); v.img.material.needsUpdate = true; }
    this.imgNote = `Ground: real Sentinel-2 true colour, ${pick[1]}${I.matches_plots ? "" : " (the plots on it are synthetic)"} · ${I.attribution}`;
  }

  setDay(day) {
    if (!this.R) return;
    const newDay = day !== this.day;
    this.day = day;
    if (newDay) {
      const today = this.v.passByDay.get(day);
      const queue = today ? SAT_ORDER.filter((k) => today.has(k) && this.show[k]) : [];
      this.done = new Set();
      for (const k of SAT_ORDER) { const S = this.v.sats[k]; S.model.visible = S.swath.visible = S.beam.visible = false; }
      this.passAnim = queue.length && this.show.signals ? { queue, i: 0, start: performance.now() } : null;
      if (!this.passAnim) { this.done = new Set(SAT_ORDER); this.setSun(null); }
    }
    this.applyDay();
  }

  applyDay() {
    const R = this.R, v = this.v, d = this.day;
    if (!R || d === null) return;
    this.applyImagery();
    const n = R.n, dummy = new THREE.Object3D(), c = new THREE.Color();
    // soil and water colour per field
    const soilCol = v.soil.geometry.attributes.color, waterCol = v.water.geometry.attributes.color;
    for (let f = 0; f < n; f++) {
      const h = this.obs(f, d);
      if (h === null) c.copy(SEASON_COLORS.bare);
      else if (h > 0) c.copy(SEASON_COLORS.wetMud);
      else if (h > -60) c.copy(SEASON_COLORS.moist).lerp(SEASON_COLORS.wetMud, 1 + h / 60);
      else c.copy(SEASON_COLORS.moist).lerp(SEASON_COLORS.dry, Math.min(1, (-h - 60) / 160));
      const [s0, cnt] = v.ranges[f];
      for (let i = s0; i < s0 + cnt; i++) soilCol.setXYZ(i, c.r, c.g, c.b);
      c.copy(SEASON_COLORS.shallow).lerp(SEASON_COLORS.deep, Math.min(1, Math.max(0, h || 0) / 70));
      for (let i = s0; i < s0 + cnt; i++) waterCol.setXYZ(i, c.r, c.g, c.b);
    }
    soilCol.needsUpdate = waterCol.needsUpdate = true;
    // rice
    // the crop (Experiment 27): height from the crop model, hill width from tillers per hill (thin seedlings to
    // bushy hills), colour from canopy cover and ripening; grain heads from heading on
    const cp = R.crop, pc = new THREE.Color();
    for (let f = 0; f < n; f++) {
      const lai = R.lai[f][d];
      let hgt, wid, head = 0, ripen = 0;
      if (cp) {
        const hcm = cp.height_cm[f][d], till = cp.tillers[f][d], stage = cp.stage[f][d];
        ripen = cp.ripen[f][d];
        hgt = stage >= 0 ? RICE_H * hcm / 100 : 0;
        wid = 0.22 + 1.28 * Math.max(0, Math.min(1, (till - 2) / 20));
        head = stage >= 3 ? 0.45 + 0.55 * Math.min(1, ripen * 4) : 0;
        c.copy(SEASON_COLORS.young).lerp(SEASON_COLORS.green, Math.min(1, cp.cover[f][d] / 0.85));
        if (ripen > 0) c.lerp(SEASON_COLORS.ripe, Math.min(1, ripen * 1.2));
        pc.copy(SEASON_COLORS.panicleGreen).lerp(SEASON_COLORS.panicleRipe, Math.min(1, ripen * 1.5));
      } else {   // an older payload: the straight-line crop
        const tau = d - R.fields[f].tp;
        hgt = lai > 0 ? RICE_H * (0.16 + 0.84 * Math.min(lai, 5) / 5) : 0;
        wid = 0.6 + 0.9 * Math.min(lai, 5) / 5;
        c.copy(SEASON_COLORS.young).lerp(SEASON_COLORS.green, Math.min(1, lai / 3.5));
        if (tau > 72) c.lerp(SEASON_COLORS.ripe, Math.min(1, (tau - 72) / 26));
      }
      const [s0, cnt] = v.hillRange[f];
      for (let k = s0; k < s0 + cnt; k++) {
        const [x, z, rot, s] = v.riceBase[k];
        dummy.position.set(x, 0, z); dummy.rotation.set(0, rot, 0);
        dummy.scale.set(wid * s, hgt * s, wid * s); dummy.updateMatrix();
        v.rice.setMatrixAt(k, dummy.matrix);
        v.rice.setColorAt(k, c);
        dummy.position.set(x, hgt * s * 0.86, z);
        dummy.scale.set(head * wid * s, head * hgt * s * 0.35 + 1e-4, head * wid * s); dummy.updateMatrix();
        v.panicles.setMatrixAt(k, dummy.matrix);
        v.panicles.setColorAt(k, pc);
      }
    }
    v.rice.instanceMatrix.needsUpdate = v.panicles.instanceMatrix.needsUpdate = true;
    if (v.rice.instanceColor) v.rice.instanceColor.needsUpdate = true;
    if (v.panicles.instanceColor) v.panicles.instanceColor.needsUpdate = true;
    // water targets, eased in the render loop
    v.targetH = R.truth.map((row, f) => this.obs(f, d) ?? -1);
    v.targetB = R.belief.h.map((row) => (row[d] === null ? -1 : row[d]));
    if (v.fresh) { v.waterH.set(v.targetH); v.beliefH.set(v.targetB); v.fresh = false; }
    // pumps, channels, inlets
    const running = v.pumps.map(() => false), toDate = v.pumps.map(() => 0);
    const zero = new THREE.Matrix4().makeScale(0, 0, 0);
    for (let f = 0; f < n; f++) {
      const t = R.fields[f].tw, mm = R.irr[f][d];
      if (mm > 0) running[t] = true;
      for (let k = 0; k <= d; k++) toDate[t] += (R.irr[f][k] * R.fields[f].area_m2) / 1000;
      v.inlets.setMatrixAt(f, mm > 0 ? v.inletMatrix[f] : zero);
    }
    v.inlets.instanceMatrix.needsUpdate = true;
    for (const ch of v.channels) ch.water.visible = running[ch.tw];
    v.pumps.forEach((pm, i) => {
      pm.spout.visible = running[i];
      updateTextSprite(pm.sign, `${pm.name}${running[i] ? " · pumping" : ""}`, `${Math.round(toDate[i]).toLocaleString()} m³ so far`);
    });
    // loggers
    for (const L of v.loggers) {
      L.g.visible = this.show.loggers;
      const r = L.lg.reading[d];
      // replay: the data-quality status of the day (app/qc.py) sets the LED: off when dead, amber with a fault
      const qc = R.mode === "replay" && R.qc ? R.qc.issues.filter((q) => q.logger === L.lg.id && q.start <= d && d <= q.end) : [];
      L.qc = qc.some((q) => q.severity === "high") ? "dead" : qc.some((q) => q.severity === "medium") ? "warn" : "ok";
      L.led.material.color.set(L.qc === "dead" ? "#3a3a3a" : L.qc === "warn" ? "#f0a020" : "#2bd66b");
      L.led.material.emissive.set(L.qc === "dead" ? "#000000" : L.qc === "warn" ? "#f0a020" : "#2bd66b");
      const fault = qc.find((q) => q.severity === "high" || q.severity === "medium");
      updateTextSprite(L.tag, `Logger ${L.lg.id}`, fault ? `QC: ${fault.title.replace(`Logger ${L.lg.id}`, "").replace(/^[: ]+/, "")}` :
        r === null ? "not installed yet" : r > 0 ? `${r.toFixed(0)} mm water` : `${(-r).toFixed(0)} mm below surface`);
    }
    // overlays
    const today = v.passByDay.get(d) || new Map();
    const flagged = new Set();
    for (const [k, m] of today) if (this.show[k]) for (const [f, p] of m) if (p.flag) flagged.add(f);
    for (let f = 0; f < n; f++) {
      const dp = R.dry_pred[f][d], dt = R.dry_true[f][d];
      const m = v.marks[f];
      m.alert.visible = this.show.signals && dp !== null && dp <= R.alert_days;
      m.alert.material = dt !== null && dt <= R.alert_days ? this.tex.alertHit : this.tex.alertMiss;
      m.doubt.visible = this.show.signals && flagged.has(f);
    }
    this.updateFrames();
    v.belief.visible = v.beliefRim.visible = this.show.belief;
    // weather and HUD
    const rain = R.rain[d];
    v.rain.visible = rain >= 1;
    v.rain.material.opacity = Math.min(0.55, 0.18 + rain / 40);
    v.rain.userData.speed = 40 + Math.min(40, rain * 2);
    const overcast = Math.min(1, rain / 15);
    this.sun.intensity = 3.2 * (this.sunFactor ?? 1) * (1 - 0.65 * overcast);
    this.hemi.intensity = 0.9 * (1 - 0.3 * overcast);
    this.sky.material.uniforms.turbidity.value = 7 + 12 * overcast;
    this.scene.fog.color.set(overcast > 0.3 ? "#aab3bb" : "#c9d6e3");
    const date = new Date(R.start_date + "T00:00:00Z"); date.setUTCDate(date.getUTCDate() + d);
    this.hud.querySelector(".hud-date").textContent = date.toLocaleDateString("en-GB", { day: "numeric", month: "long", timeZone: "UTC" });
    this.hud.querySelector(".hud-wx").textContent = `${rain >= 0.5 ? `Rain ${rain.toFixed(1)} mm` : "No rain"} · ET₀ ${R.et0[d].toFixed(1)} mm`;
    this.hud.querySelector(".hud-sat").innerHTML = SAT_ORDER.filter((k) => today.has(k)).map((k) => {
      const m = today.get(k), w = [...m.values()].filter((p) => isWrong(p)).length, meta = R.sensors ? R.sensors[k] : null;
      return `<div><span class="dot" style="background:${SATS[k].color}"></span>${SATS[k].label} ${SATS[k].time} (${SATS[k].band}): ${m.size} fields seen, ${w} wrong` +
        `${k === "nisar" && meta && meta.answer_update ? ` · updated ${[...m.values()].filter((p) => p.update).length} fields' advice` : meta && !meta.model_input ? " · shown only" : ""}</div>`;
    }).join("");
    this.hud.querySelector(".hud-site").textContent = (R.mode === "replay" ? `Replay: ${R.campaign.label} · fields without a logger show the model's estimate · ` : "") +
      `${R.layout.label}${R.layout.surveyed ? " · surveyed" : ""}` + (this.imgNote ? ` · ${this.imgNote}` : "");
  }

  // rings for each sensor: shown once its swath has passed the field (all of them once the day's passes are done)
  // the level drawn for a field: the truth (simulation) or its logger (replay); in replay, fields without a
  // logger show the model's estimate
  obs(f, d) {
    const R = this.R, t = R.truth[f][d];
    if (t !== null) return t;
    return R.mode === "replay" && R.planted[f][d] ? R.belief.h[f][d] : null;
  }

  updateFrames() {
    const v = this.v, R = this.R, today = v.passByDay.get(this.day) || new Map(), c = new THREE.Color();
    const pa = this.passAnim, active = pa ? pa.queue[pa.i] : null;
    for (const k of SAT_ORDER) {
      const mesh = v.frames[k], col = mesh.geometry.attributes.color, ranges = mesh.geometry.userData.ranges;
      const passes = today.get(k) || new Map();
      const shown = this.show.signals && this.show[k] && passes.size > 0;
      const done = this.done ? this.done.has(k) : true;
      const front = this.front ?? Infinity;
      for (let f = 0; f < R.n; f++) {
        const p = passes.get(f);
        let on = shown && !!p && (done || (k === active && (SATS[k].desc ? v.fields[f].c[1] < front : v.fields[f].c[1] > front)));
        c.set(p && isWrong(p) ? "#ff4d42" : SATS[k].color);
        const [s0, cnt] = ranges[f];
        for (let i = s0; i < s0 + cnt; i++) col.setXYZW(i, c.r, c.g, c.b, on ? 0.95 : 0);
      }
      col.needsUpdate = true;
    }
  }

  // sun for the time of an overpass (null: the default late-morning sun)
  setSun(k) {
    const [elev, az, color, f] = k ? SATS[k].sun : [36, 150, "#fff1d6", 1];
    const dir = new THREE.Vector3().setFromSphericalCoords(1, THREE.MathUtils.degToRad(90 - elev), THREE.MathUtils.degToRad(az));
    this.sky.material.uniforms.sunPosition.value.copy(dir);
    this.sun.color.set(color);
    this.sunFactor = f;
    if (this.v) {
      const half = Math.max(this.v.W, this.v.D) * 0.75 + 30;
      this.sun.position.copy(this.v.center).addScaledVector(dir, half + 200);
    }
    if (this.R && this.day !== null) {
      const overcast = Math.min(1, this.R.rain[this.day] / 15);
      this.sun.intensity = 3.2 * f * (1 - 0.65 * overcast);
    }
  }

  setSelected(f) {
    if (!this.v) return;
    this.selected = f;
    if (this.v.sel) { this.v.group.remove(this.v.sel); this.v.sel.geometry.dispose(); this.v.sel = null; }
    if (f === null || f === undefined || f >= this.R.n) return;
    this.v.sel = new THREE.Mesh(ribbonGeometry([insetPoly(this.v.fields[f].pts, -0.35)], 0.5, BUND_H + 0.1, true),
      new THREE.MeshBasicMaterial({ color: "#2a78d6", side: THREE.DoubleSide }));
    this.v.group.add(this.v.sel);
  }

  setVisible(on) {
    this.visible = on;
    this.container.hidden = !on;
    if (on) { this.resize(); this.clock.getDelta(); }
  }

  resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  // ── animation ─────────────────────────────────────────────────────────
  loop() {
    requestAnimationFrame(this.loop);
    if (!this.visible || !this.v) return;
    const dt = Math.min(0.05, this.clock.getDelta());
    this.uTime.value += dt;
    const v = this.v;
    if (v.targetH) {
      const ease = Math.min(1, dt * 5);
      const wp = v.water.geometry.attributes.position, bp = v.belief.geometry.attributes.position, rp = v.beliefRim.geometry.attributes.position;
      const rimRanges = v.beliefRim.geometry.userData.ranges;
      for (let f = 0; f < this.R.n; f++) {
        v.waterH[f] += (v.targetH[f] - v.waterH[f]) * ease;
        v.beliefH[f] += (v.targetB[f] - v.beliefH[f]) * ease;
        const h = v.waterH[f], b = v.beliefH[f];
        const yw = h > 0.5 ? h * VY + 0.005 : -0.5;
        const yb = v.targetB[f] < -0.5 && v.targetH[f] < -0.5 ? -0.5 : (b > 0 ? b * VY : 0) + 0.03;
        const [s0, cnt] = v.ranges[f];
        for (let i = s0; i < s0 + cnt; i++) { wp.setY(i, yw); bp.setY(i, yb); }
        const [r0, rc] = rimRanges[f];
        for (let i = r0; i < r0 + rc; i++) rp.setY(i, yb + 0.01);
      }
      wp.needsUpdate = bp.needsUpdate = rp.needsUpdate = true;
      v.water.geometry.computeBoundingSphere();
    }
    this.tex.ripple.offset.x += dt * 0.01; this.tex.ripple.offset.y += dt * 0.006;
    this.tex.flow.offset.x -= dt * 0.6;
    v.inlets.material.opacity = 0.75 + 0.2 * Math.sin(this.uTime.value * 8);
    for (const pm of v.pumps) if (pm.spout.visible) animateSpout(pm.spout, this.uTime.value);
    for (const L of v.loggers) L.led.material.emissiveIntensity = L.qc === "dead" ? 0 : Math.sin(this.uTime.value * (L.qc === "warn" ? 9 : 3) + L.g.position.x) > 0.6 ? 2.5 : 0.2;
    if (v.rain.visible) animateRain(v.rain, dt);
    if (this.passAnim) {   // satellites cross one after another, in time-of-day order (wall clock: same speed on slow GPUs)
      const pa = this.passAnim, k = pa.queue[pa.i], S = v.sats[k];
      if (pa.lit !== pa.i) { this.setSun(k); pa.lit = pa.i; }
      const t = this.debugPassT ?? Math.min(1, (performance.now() - pa.start) / 2600);   // debugPassT: freeze a pass (screenshots)
      const along = v.bbox.z0 - 20 + t * (v.D + 40);
      this.front = SATS[k].desc ? along : v.bbox.z1 + v.bbox.z0 - along;
      S.model.visible = S.swath.visible = S.beam.visible = t < 1;
      // drawn low (22 units, not to scale) so the pass reads from the usual viewpoint; the beam joins it to its swath
      const sx = v.center.x + (k === "s2" ? -8 : -24), sz = this.front + (SATS[k].desc ? -6 : 6);
      S.model.position.set(sx, 22, sz);
      S.model.rotation.y = SATS[k].desc ? 0 : Math.PI;
      const swX = v.center.x + (k === "s2" ? 0 : 12);   // SAR looks to the side of its track
      S.swath.position.set(swX, BUND_H + 0.4, this.front);
      const bp = S.beam.geometry.attributes.position, hw = v.W * 0.8;
      bp.setXYZ(0, sx, 21, sz); bp.setXYZ(1, swX - hw, BUND_H + 0.4, this.front); bp.setXYZ(2, swX + hw, BUND_H + 0.4, this.front);
      bp.needsUpdate = true;
      this.updateFrames();
      if (t >= 1) {
        this.done.add(k);
        S.model.visible = S.swath.visible = S.beam.visible = false;
        pa.i += 1; pa.start = performance.now();
        if (pa.i >= pa.queue.length) { this.passAnim = null; this.done = new Set(SAT_ORDER); this.front = null; this.setSun(null); this.updateFrames(); }
      }
    }
    if (v.sel) v.sel.material.color.setHSL(0.59, 0.68, 0.45 + 0.12 * Math.sin(this.uTime.value * 4));
    if (this.flight) {
      const fl = this.flight;
      fl.t = Math.min(1, fl.t + dt / 1.2);
      const e = 1 - Math.pow(1 - fl.t, 3);
      this.camera.position.lerpVectors(fl.fromPos, fl.toPos, e);
      this.controls.target.lerpVectors(fl.fromT, fl.toT, e);
      if (fl.t >= 1) this.flight = null;
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}

// ── polygon helpers (scene units: x east, z south) ─────────────────────────
// merged polygon surfaces at height y; per-field vertex ranges in userData.ranges
function polyGeometry(fields, tris, y, withUv) {
  const pos = [], uv = [], idx = [], ranges = [];
  fields.forEach((fl, f) => {
    const base = pos.length / 3;
    ranges.push([base, fl.pts.length]);
    for (const [x, z] of fl.pts) { pos.push(x, y, z); uv.push(x / 8, z / 8); }
    for (const [a, b, c] of tris[f]) idx.push(base + a, base + c, base + b);   // wound to face up
  });
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("normal", new THREE.Float32BufferAttribute(new Float32Array(pos.length).map((_, i) => (i % 3 === 1 ? 1 : 0)), 3));
  g.setAttribute("color", new THREE.Float32BufferAttribute(new Float32Array(pos.length).fill(1), 3));
  if (withUv) g.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
  g.setIndex(idx);
  g.userData.ranges = ranges;
  return g;
}

// flat ribbons along polylines (closed: polygon outlines); rgba: per-vertex RGBA colour for visibility
function ribbonGeometry(lines, width, y, closed, rgba = false) {
  const pos = [], uv = [], idx = [], ranges = [];
  for (const pts of lines) {
    const base = pos.length / 3, m = pts.length;
    ranges.push([base, 2 * m]);
    let along = 0;
    for (let i = 0; i < m; i++) {
      const prev = pts[closed ? (i - 1 + m) % m : Math.max(0, i - 1)], next = pts[closed ? (i + 1) % m : Math.min(m - 1, i + 1)];
      let dx = next[0] - prev[0], dz = next[1] - prev[1];
      const l = Math.hypot(dx, dz) || 1; dx /= l; dz /= l;
      const nx = -dz * width / 2, nz = dx * width / 2;
      if (i > 0) along += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
      pos.push(pts[i][0] + nx, y, pts[i][1] + nz, pts[i][0] - nx, y, pts[i][1] - nz);
      uv.push(along / 3, 0, along / 3, 1);
    }
    const segs = closed ? m : m - 1;
    for (let i = 0; i < segs; i++) {
      const a = base + 2 * i, b = base + 2 * ((i + 1) % m);
      idx.push(a, b, a + 1, a + 1, b, b + 1);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
  if (rgba) g.setAttribute("color", new THREE.Float32BufferAttribute(new Float32Array((pos.length / 3) * 4), 4));
  g.setIndex(idx);
  g.computeVertexNormals();
  g.userData.ranges = ranges;
  return g;
}

function insetPoly(pts, d) {   // move each vertex towards the centroid by d (negative: outwards)
  const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length, cz = pts.reduce((s, p) => s + p[1], 0) / pts.length;
  return pts.map(([x, z]) => { const l = Math.hypot(cx - x, cz - z) || 1; return [x + ((cx - x) / l) * d, z + ((cz - z) / l) * d]; });
}
function offsetLine(pts, d) {
  return pts.map((p, i) => {
    const a = pts[Math.max(0, i - 1)], b = pts[Math.min(pts.length - 1, i + 1)];
    const dx = b[0] - a[0], dz = b[1] - a[1], l = Math.hypot(dx, dz) || 1;
    return [p[0] - (dz / l) * d, p[1] + (dx / l) * d];
  });
}
function inPolygon(x, z, pts) {
  let c = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++)
    if ((pts[i][1] > z) !== (pts[j][1] > z) && x < pts[j][0] + ((z - pts[j][1]) * (pts[i][0] - pts[j][0])) / (pts[i][1] - pts[j][1])) c = !c;
  return c;
}
function segDist(x, z, a, b) {
  const dx = b[0] - a[0], dz = b[1] - a[1], l2 = dx * dx + dz * dz || 1;
  const t = Math.max(0, Math.min(1, ((x - a[0]) * dx + (z - a[1]) * dz) / l2));
  return Math.hypot(x - (a[0] + t * dx), z - (a[1] + t * dz));
}
function edgeDist(x, z, pts) {
  let d = Infinity;
  for (let i = 0; i < pts.length; i++) d = Math.min(d, segDist(x, z, pts[i], pts[(i + 1) % pts.length]));
  return d;
}
// rice hills on a grid aligned with the plot's longest side, kept off the bunds; at most ~420 per plot
function hillPositions(pts, rng) {
  let best = 0, ang = 0;
  pts.forEach((a, i) => { const b = pts[(i + 1) % pts.length], l = Math.hypot(b[0] - a[0], b[1] - a[1]); if (l > best) { best = l; ang = Math.atan2(b[1] - a[1], b[0] - a[0]); } });
  const ux = Math.cos(ang), uz = Math.sin(ang), area = Math.abs(pts.reduce((s, a, i) => { const b = pts[(i + 1) % pts.length]; return s + a[0] * b[1] - b[0] * a[1]; }, 0)) / 2;
  const sp = Math.max(1.1, Math.sqrt(area / 420));
  const us = pts.map(([x, z]) => x * ux + z * uz), vs = pts.map(([x, z]) => -x * uz + z * ux);
  const out = [];
  for (let u = Math.min(...us) + sp / 2; u < Math.max(...us); u += sp)
    for (let w = Math.min(...vs) + sp / 2; w < Math.max(...vs); w += sp) {
      const x = u * ux - w * uz + (rng() - 0.5) * 0.25, z = u * uz + w * ux + (rng() - 0.5) * 0.25;
      if (inPolygon(x, z, pts) && edgeDist(x, z, pts) > BUND / 2 + 0.35) out.push([x, z]);
    }
  return out;
}
// the point on a plot's boundary closest to a polyline (its channel), and that edge's direction
function nearestEdgePoint(pts, line) {
  let best = { d: Infinity, p: pts[0], ang: 0 };
  for (let i = 0; i < pts.length; i++) {
    const a = pts[i], b = pts[(i + 1) % pts.length];
    for (let k = 0; k <= 8; k++) {
      const t = 0.1 + 0.8 * (k / 8), x = a[0] + t * (b[0] - a[0]), z = a[1] + t * (b[1] - a[1]);
      let d = Infinity;
      for (let j = 0; j < line.length; j++) d = Math.min(d, line.length > 1 && j < line.length - 1 ? segDist(x, z, line[j], line[j + 1]) : Math.hypot(x - line[j][0], z - line[j][1]));
      if (d < best.d) best = { d, p: [x, z], ang: Math.atan2(b[1] - a[1], b[0] - a[0]) };
    }
  }
  return best;
}

function loggerGeometries() {
  return { pipe: new THREE.CylinderGeometry(0.11, 0.11, 1.9, 12), head: new THREE.BoxGeometry(0.42, 0.32, 0.42),
           panel: new THREE.BoxGeometry(0.5, 0.04, 0.38), antenna: new THREE.CylinderGeometry(0.018, 0.018, 0.9, 6),
           led: new THREE.SphereGeometry(0.05, 8, 8) };
}
function loggerMaterials() {
  return { pipe: new THREE.MeshStandardMaterial({ color: "#eef1f2", roughness: 0.5 }),
           head: new THREE.MeshStandardMaterial({ color: "#5f6b73", roughness: 0.4, metalness: 0.4 }),
           panel: new THREE.MeshStandardMaterial({ color: "#1d3a6b", roughness: 0.3, metalness: 0.4 }),
           led: new THREE.MeshStandardMaterial({ color: "#2bd66b", emissive: "#2bd66b", emissiveIntensity: 1 }) };
}

// trees and homesteads around the fields (never on a plot)
function surroundingsAround(group, rng, v) {
  const { x0, x1, z0, z1 } = v.bbox;
  const onField = (x, z, r) => x > x0 - r && x < x1 + r && z > z0 - r && z < z1 + r &&
    v.fields.some((fl) => Math.abs(fl.c[0] - x) < 40 && Math.abs(fl.c[1] - z) < 40 && (inPolygon(x, z, fl.pts) || edgeDist(x, z, fl.pts) < r));
  const spots = [];
  for (let i = 0; i < 600 && spots.length < 280; i++) {
    const side = rng();
    let x, z;
    if (side < 0.35) { x = x0 - 12 - rng() * 45; z = z0 - 10 + rng() * (z1 - z0 + 20); }
    else if (side < 0.55) { x = x1 + 5 + rng() * 32; z = z0 - 10 + rng() * (z1 - z0 + 20); }
    else if (side < 0.77) { x = x0 - 10 + rng() * (x1 - x0 + 20); z = z0 - 5 - rng() * 32; }
    else { x = x0 - 10 + rng() * (x1 - x0 + 20); z = z1 + 5 + rng() * 32; }
    if (!onField(x, z, 3.5)) spots.push([x, z, 0.8 + rng() * 0.7, rng()]);
  }
  const trunk = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.18, 0.28, 3, 6), new THREE.MeshStandardMaterial({ color: "#5b4632", roughness: 1 }), spots.length);
  const crown = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(2.2, 1), new THREE.MeshStandardMaterial({ roughness: 0.95, flatShading: true }), spots.length);
  const dummy = new THREE.Object3D(), c = new THREE.Color();
  spots.forEach(([x, z, s, hue], i) => {
    dummy.position.set(x, 1.5 * s, z); dummy.scale.set(s, s, s); dummy.rotation.set(0, 0, 0); dummy.updateMatrix(); trunk.setMatrixAt(i, dummy.matrix);
    dummy.position.set(x, 3.6 * s, z); dummy.scale.set(s * (1 + hue * 0.4), s * (1.1 + hue * 0.5), s * (1 + hue * 0.3)); dummy.rotation.set(0, hue * 6, 0); dummy.updateMatrix();
    crown.setMatrixAt(i, dummy.matrix);
    crown.setColorAt(i, c.setHSL(0.24 + hue * 0.08, 0.45, 0.22 + hue * 0.1));
  });
  trunk.castShadow = crown.castShadow = true;
  group.add(trunk, crown);
  const wallMat = new THREE.MeshStandardMaterial({ roughness: 0.95 });
  const roofMat = new THREE.MeshStandardMaterial({ color: "#8d9aa3", metalness: 0.55, roughness: 0.45 });
  for (let i = 0; i < 12; i++) {
    const h = new THREE.Group(), w = 3 + rng() * 2, d = 2.5 + rng() * 1.5;
    const walls = new THREE.Mesh(new THREE.BoxGeometry(w, 2, d), wallMat.clone());
    walls.material.color.setHSL(0.08 + rng() * 0.05, 0.25 + rng() * 0.2, 0.55 + rng() * 0.2);
    walls.position.y = 1;
    const roof = new THREE.Mesh(new THREE.CylinderGeometry(d * 0.62, d * 0.62, w + 0.4, 3, 1), roofMat);
    roof.rotation.z = Math.PI / 2; roof.rotation.x = Math.PI / 6; roof.position.y = 2.35; roof.scale.set(1, 1, 0.55);
    for (const m of [walls, roof]) { m.castShadow = m.receiveShadow = true; h.add(m); }
    h.position.set(x0 - 30 - rng() * 25, 0, z0 + 6 + rng() * (z1 - z0 - 12));
    h.rotation.y = rng() * Math.PI;
    group.add(h);
  }
}

// ── geometry helpers ──────────────────────────────────────────────────────
function flatPlane(w, h) {
  const g = new THREE.PlaneGeometry(w, h);
  g.rotateX(-Math.PI / 2);
  return g;
}

// grain heads of one hill: ~8 arching, drooping panicles of height ~1 (scaled per instance, placed at the hill's top)
function panicleGeometry() {
  const pos = [], col = [], idx = [];
  const rng = mulberry(7), P = 8, SEG = 4;
  for (let b = 0; b < P; b++) {
    const ang = (b / P) * Math.PI * 2 + rng() * 0.6, reach = 0.25 + rng() * 0.2, w0 = 0.07;
    const dx = Math.cos(ang), dz = Math.sin(ang), px = -dz, pz = dx, base = pos.length / 3;
    for (let s_ = 0; s_ <= SEG; s_++) {
      const t = s_ / SEG, r = 0.05 + reach * t, y = 0.6 * t - 1.1 * t * t * 0.5, w = w0 * (0.6 + 0.4 * Math.sin(Math.PI * t));
      pos.push(dx * r + px * w, y, dz * r + pz * w, dx * r - px * w, y, dz * r - pz * w);
      col.push(1, 1, 1, 1, 1, 1);
      if (s_ < SEG) { const a = base + s_ * 2; idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); }
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
  g.setIndex(idx);
  g.computeVertexNormals();
  return g;
}

// one rice hill: 18 curved blades, height 1 (scaled per instance; its width follows the tillers); darker at the base
function riceHillGeometry() {
  const pos = [], col = [], idx = [];
  const rng = mulberry(42), B = 18, SEG = 3;
  for (let b = 0; b < B; b++) {
    const ang = (b / B) * Math.PI * 2 + rng() * 0.5, lean = 0.18 + rng() * 0.32, top = 0.75 + rng() * 0.3, w0 = 0.06 + rng() * 0.03;
    const dx = Math.cos(ang), dz = Math.sin(ang), px = -dz, pz = dx;
    const base = pos.length / 3;
    for (let s = 0; s <= SEG; s++) {
      const t = s / SEG, r = 0.04 + lean * Math.pow(t, 1.6), y = top * (t - 0.12 * t * t * lean * 3), w = w0 * (1 - 0.85 * t);
      pos.push(dx * r + px * w, y, dz * r + pz * w, dx * r - px * w, y, dz * r - pz * w);
      const shade = 0.62 + 0.38 * t;
      col.push(shade, shade, shade, shade, shade, shade);
      if (s < SEG) { const a = base + s * 2; idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); }
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
  g.setIndex(idx);
  g.computeVertexNormals();
  const n = g.attributes.normal;
  for (let i = 0; i < n.count; i++) { n.setY(i, Math.abs(n.getY(i)) + 0.8); }  // grass-like soft lighting
  g.normalizeNormals();
  return g;
}

function pumpHouse(tex) {
  const g = new THREE.Group();
  const brick = new THREE.MeshStandardMaterial({ map: tex.brick, roughness: 0.9 });
  const tin = new THREE.MeshStandardMaterial({ color: "#7d8c96", roughness: 0.45, metalness: 0.6 });
  const slab = new THREE.Mesh(new THREE.BoxGeometry(5.2, 0.3, 4.4), new THREE.MeshStandardMaterial({ color: "#9a9890", roughness: 0.9 }));
  slab.position.y = 0.0;
  const walls = new THREE.Mesh(new THREE.BoxGeometry(3.2, 2.4, 2.8), brick);
  walls.position.set(-0.4, 1.35, 0);
  const roof = new THREE.Mesh(new THREE.BoxGeometry(3.8, 0.12, 3.4), tin);
  roof.position.set(-0.4, 2.7, 0); roof.rotation.z = 0.12;
  const door = new THREE.Mesh(new THREE.BoxGeometry(0.05, 1.6, 0.9), new THREE.MeshStandardMaterial({ color: "#2f4a5e", roughness: 0.6, metalness: 0.3 }));
  door.position.set(1.21, 0.95, 0.6);
  const pipe = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.2, 2.2, 16), new THREE.MeshStandardMaterial({ color: "#3d4a52", roughness: 0.4, metalness: 0.7 }));
  pipe.rotation.z = Math.PI / 2; pipe.position.set(1.9, 0.9, 0);
  const basin = new THREE.Mesh(new THREE.BoxGeometry(1.6, 0.5, 2.4), new THREE.MeshStandardMaterial({ color: "#8f8c84", roughness: 0.95 }));
  basin.position.set(2.6, 0.1, 0);
  const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.09, 6, 8), new THREE.MeshStandardMaterial({ color: "#6b6b66" }));
  pole.position.set(-2.3, 3, 1.8);
  for (const m of [slab, walls, roof, door, pipe, basin, pole]) { m.castShadow = true; m.receiveShadow = true; g.add(m); }
  return g;
}

function spoutParticles() {
  const N = 140, geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(N * 3), 3));
  const pts = new THREE.Points(geo, new THREE.PointsMaterial({ color: "#d8f0ff", size: 0.16, transparent: true, opacity: 0.85, depthWrite: false }));
  pts.userData.seed = Array.from({ length: N }, () => [Math.random(), (Math.random() - 0.5) * 0.3, (Math.random() - 0.5) * 0.3]);
  pts.visible = false;
  return pts;
}
function animateSpout(pts, time) {
  const a = pts.geometry.attributes.position;
  pts.userData.seed.forEach(([ph, jy, jz], i) => {
    const t = (time * 1.6 + ph) % 1;
    a.setXYZ(i, t * 2.2, 0.2 + jy * 0.4 + 0.7 * t - 2.4 * t * t, jz * (1 + t));
  });
  a.needsUpdate = true;
}

function rainLines(w, d, cx, cz) {
  const N = 4500, pos = new Float32Array(N * 6), H = 60;
  for (let i = 0; i < N; i++) {
    const x = cx + (Math.random() - 0.5) * w, z = cz + (Math.random() - 0.5) * d, y = Math.random() * H;
    pos.set([x, y, z, x + 0.05, y + 1.2, z + 0.1], i * 6);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  const l = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: "#c8d6e2", transparent: true, opacity: 0.4 }));
  l.userData = { H, speed: 50 };
  l.visible = false;
  return l;
}
function animateRain(l, dt) {
  const a = l.geometry.attributes.position.array, dy = l.userData.speed * dt, H = l.userData.H;
  for (let i = 0; i < a.length; i += 6) {
    a[i + 1] -= dy; a[i + 4] -= dy;
    if (a[i + 1] < 0) { a[i + 1] += H; a[i + 4] += H; }
  }
  l.geometry.attributes.position.needsUpdate = true;
}

// stylised spacecraft: Sentinel-2 (one solar wing, multispectral imager), Sentinel-1 (two wings and a 12 m
// SAR antenna along track), NISAR (two wings, a boom and the 12 m L-band mesh reflector)
function satelliteModel(k) {
  const g = new THREE.Group();
  const foil = new THREE.MeshStandardMaterial({ color: "#d8b25a", metalness: 0.8, roughness: 0.35 });
  const white = new THREE.MeshStandardMaterial({ color: "#e4e4df", metalness: 0.3, roughness: 0.5 });
  const cells = new THREE.MeshStandardMaterial({ color: "#1d3a6b", metalness: 0.4, roughness: 0.3, emissive: "#0c1c38" });
  const dark = new THREE.MeshStandardMaterial({ color: "#3b4148", metalness: 0.5, roughness: 0.4 });
  const add = (geo, mat, x, y, z, rx = 0, ry = 0, rz = 0) => { const m = new THREE.Mesh(geo, mat); m.position.set(x, y, z); m.rotation.set(rx, ry, rz); g.add(m); return m; };
  if (k === "s2") {
    add(new THREE.BoxGeometry(2.2, 2.8, 2.2), foil, 0, 0, 0);
    add(new THREE.BoxGeometry(9, 0.08, 2.4), cells, 5.8, 0.6, 0);
    add(new THREE.BoxGeometry(1.2, 0.9, 1.2), dark, 0, -1.8, 0);          // multispectral instrument, looking down
  } else if (k === "s1") {
    add(new THREE.BoxGeometry(2.4, 2.4, 3), white, 0, 0, 0);
    for (const s of [-1, 1]) add(new THREE.BoxGeometry(7.5, 0.08, 2.4), cells, s * 5, 0.4, 0);
    add(new THREE.BoxGeometry(1.0, 0.18, 12), dark, -1.6, -1.6, 0, 0, 0, -0.35); // C-band SAR antenna, side-looking
  } else {
    add(new THREE.BoxGeometry(2.6, 3, 2.6), foil, 0, 0, 0);
    for (const s of [-1, 1]) add(new THREE.BoxGeometry(6.5, 0.08, 2.6), cells, s * 4.6, 0.6, 0);
    add(new THREE.CylinderGeometry(0.08, 0.08, 7, 6), white, -2.2, -2.6, 0, 0, 0, 0.9);                       // boom
    const dish = add(new THREE.SphereGeometry(6, 24, 8, 0, Math.PI * 2, 0, 0.55), new THREE.MeshStandardMaterial({
      color: "#e8c766", metalness: 0.7, roughness: 0.4, side: THREE.DoubleSide, wireframe: true }), -5, -6.4, 0, Math.PI + 0.5, 0, 0);
    dish.scale.set(1, 0.5, 1);                                                                                  // L-band mesh reflector
  }
  g.scale.setScalar(1.15);
  return g;
}

function disposeGroup(g) {
  g.traverse((o) => {
    if (o.geometry) o.geometry.dispose();
    if (o.material && !o.isSprite) (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => m.dispose());
  });
}

// ── textures (procedural, so the app needs no image files) ───────────────────
function canvasTex(size, draw, repeat = 1) {
  const cv = document.createElement("canvas");
  cv.width = cv.height = size;
  draw(cv.getContext("2d"), size);
  const t = new THREE.CanvasTexture(cv);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(repeat, repeat);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  return t;
}
function noise(ctx, s, base, amp, n = 6000, r = 2) {
  ctx.fillStyle = base; ctx.fillRect(0, 0, s, s);
  for (let i = 0; i < n; i++) {
    const v = Math.floor((Math.random() - 0.5) * amp);
    ctx.fillStyle = `rgba(${v > 0 ? 255 : 0},${v > 0 ? 255 : 0},${v > 0 ? 255 : 0},${Math.abs(v) / 255})`;
    ctx.fillRect(Math.random() * s, Math.random() * s, r * Math.random() + 0.5, r * Math.random() + 0.5);
  }
}
function makeTextures() {
  const tex = {};
  tex.grass = canvasTex(256, (c, s) => { noise(c, s, "#5d7a35", 70, 9000, 3); }, 300);
  tex.dirt = canvasTex(256, (c, s) => { noise(c, s, "#8c7556", 60, 8000, 2); }, 40);
  tex.bund = canvasTex(128, (c, s) => {
    noise(c, s, "#6f7a3c", 80, 3000, 2);
    for (let i = 0; i < 400; i++) { c.fillStyle = `rgba(110,85,55,${Math.random() * 0.5})`; c.fillRect(Math.random() * s, Math.random() * s, 3, 3); }
  }, 1);
  tex.soil = canvasTex(256, (c, s) => {
    noise(c, s, "#e9e6e1", 40, 6000, 2);
    c.strokeStyle = "rgba(90,75,60,0.32)"; c.lineWidth = 1;
    for (let i = 0; i < 45; i++) {                                         // dry-soil crack network
      let x = Math.random() * s, y = Math.random() * s; c.beginPath(); c.moveTo(x, y);
      for (let k = 0; k < 6; k++) { x += (Math.random() - 0.5) * 50; y += (Math.random() - 0.5) * 50; c.lineTo(x, y); }
      c.stroke();
    }
  }, 1);
  tex.brick = canvasTex(128, (c, s) => {
    c.fillStyle = "#c8c2b6"; c.fillRect(0, 0, s, s);
    for (let y = 0; y < s; y += 16) for (let x = (y / 16) % 2 ? -16 : 0; x < s; x += 32) {
      c.fillStyle = `hsl(${12 + Math.random() * 8},${40 + Math.random() * 15}%,${36 + Math.random() * 10}%)`;
      c.fillRect(x + 1, y + 1, 30, 14);
    }
  }, 1);
  tex.flow = canvasTex(128, (c, s) => {
    c.fillStyle = "#6f97ad"; c.fillRect(0, 0, s, s);
    for (let i = 0; i < 90; i++) { c.fillStyle = `rgba(230,245,255,${0.15 + Math.random() * 0.35})`; c.fillRect(Math.random() * s, Math.random() * s, 10 + Math.random() * 30, 1.5); }
  }, 1);
  tex.flow.repeat.set(12, 1);
  // ripple normal map from a sum of waves
  const rs = 128, cv = document.createElement("canvas"); cv.width = cv.height = rs;
  const ctx = cv.getContext("2d"), img = ctx.createImageData(rs, rs);
  const hgt = (x, y) => Math.sin(x * 0.29 + y * 0.11) + 0.6 * Math.sin(x * 0.13 - y * 0.37) + 0.35 * Math.sin((x + y) * 0.61);
  for (let y = 0; y < rs; y++) for (let x = 0; x < rs; x++) {
    const nx = hgt(x - 1, y) - hgt(x + 1, y), ny = hgt(x, y - 1) - hgt(x, y + 1), len = Math.hypot(nx, ny, 2);
    const i = (y * rs + x) * 4;
    img.data[i] = (nx / len * 0.5 + 0.5) * 255; img.data[i + 1] = (ny / len * 0.5 + 0.5) * 255; img.data[i + 2] = (2 / len * 0.5 + 0.5) * 255; img.data[i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  tex.ripple = new THREE.CanvasTexture(cv);
  tex.ripple.wrapS = tex.ripple.wrapT = THREE.RepeatWrapping;
  tex.ripple.repeat.set(1.5, 1.5);
  // markers
  const marker = (draw) => new THREE.SpriteMaterial({ map: canvasTex(128, draw), depthWrite: false, transparent: true });
  const drop = (c, fill) => {
    c.beginPath(); c.moveTo(64, 14); c.bezierCurveTo(98, 60, 104, 80, 104, 88); c.arc(64, 88, 40, 0, Math.PI); c.bezierCurveTo(24, 80, 30, 60, 64, 14);
    c.lineWidth = 10; c.strokeStyle = "#d99a00"; c.fillStyle = "#f0b323";
    if (fill) c.fill(); else { c.fillStyle = "rgba(255,255,255,0.85)"; c.fill(); }
    c.stroke();
  };
  tex.alertHit = marker((c) => drop(c, true));
  tex.alertMiss = marker((c) => drop(c, false));
  tex.doubt = marker((c) => {
    c.beginPath(); c.arc(64, 64, 52, 0, Math.PI * 2); c.fillStyle = "#e0662a"; c.fill();
    c.fillStyle = "#fff"; c.font = "bold 84px system-ui, sans-serif"; c.textAlign = "center"; c.textBaseline = "middle"; c.fillText("!", 64, 70);
  });
  return tex;
}

function textSprite(title, sub) {
  const cv = document.createElement("canvas"); cv.width = 512; cv.height = 128;
  const t = new THREE.CanvasTexture(cv); t.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: t, depthWrite: false, transparent: true }));
  s.scale.set(10, 2.5, 1);
  s.userData.canvas = cv;
  updateTextSprite(s, title, sub);
  return s;
}
function updateTextSprite(s, title, sub) {
  const key = title + "|" + sub;
  if (s.userData.key === key) return;
  s.userData.key = key;
  const c = s.userData.canvas.getContext("2d");
  c.clearRect(0, 0, 512, 128);
  c.fillStyle = "rgba(20,24,28,0.72)";
  c.beginPath(); c.roundRect(4, 8, 504, 112, 22); c.fill();
  c.fillStyle = "#fff"; c.font = "600 38px system-ui, sans-serif"; c.textAlign = "center"; c.fillText(title, 256, 54, 480);
  c.fillStyle = "#cfd8df"; c.font = "36px system-ui, sans-serif"; c.fillText(sub, 256, 100);
  s.material.map.needsUpdate = true;
}

function mulberry(a) {
  return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
