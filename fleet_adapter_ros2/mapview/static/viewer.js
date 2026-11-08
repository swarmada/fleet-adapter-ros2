// Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
// SPDX-License-Identifier: Apache-2.0
//
// The browser viewer of the map-view contract, mapview/0 (CONTRACT.md beside the service).
// Read-only: it opens one server-sent-event stream from the service that served it and sends
// nothing anywhere. Every message is checked as the contract says before it is drawn.
"use strict";

const VERSION = "mapview/0";
const POSE_STALE_S = 2, POSE_LOST_S = 10, MAP_STALE_S = 35;
const MAX_CELLS = 4096 * 4096, PATH_STEP_M = 0.02, MAX_TEXT = 200;
const REQUIRED = {
  MapSnapshot: ["seq", "resolution", "origin", "width", "height", "cells", "sha256"],
  PoseSample: ["x", "y", "yaw_deg", "source"],
  Overlays: ["keepouts", "zones", "target"],
  SessionStatus: ["source", "phase", "text", "area_free_m2"],
};
const ROBOT_RE = /^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$/;
const FRAME_RE = /^[A-Za-z0-9_@./-]{1,128}$/;
const SHA_RE = /^[0-9a-f]{64}$/;

class Refused extends Error {
  constructor(why, detail) { super(detail); this.why = why; }
}

const num = (v) => typeof v === "number" && Number.isFinite(v);
const int = (v, lo) => Number.isInteger(v) && v >= lo;

function checkGridHeader(f, where) {
  if (!f || typeof f !== "object") throw new Refused("contract", `${where}: grid`);
  if (!num(f.resolution) || f.resolution <= 0) throw new Refused("contract", `${where}: resolution`);
  if (!Array.isArray(f.origin) || f.origin.length !== 2 || !f.origin.every(num))
    throw new Refused("contract", `${where}: origin`);
  if (!int(f.width, 1) || !int(f.height, 1) || f.width * f.height > MAX_CELLS)
    throw new Refused("contract", `${where}: size`);
  if (typeof f.cells !== "string" || typeof f.sha256 !== "string" || !SHA_RE.test(f.sha256))
    throw new Refused("contract", `${where}: cells`);
}

// The contract's checks, in its order: the version first, and nothing else read without it.
function check(m) {
  if (!m || typeof m !== "object" || Array.isArray(m)) throw new Refused("contract", "not an object");
  if (m.contractVersion !== VERSION) throw new Refused("version", String(m.contractVersion).slice(0, 32));
  for (const k of ["kind", "robot", "frame", "stamp"])
    if (!(k in m)) throw new Refused("contract", `missing ${k}`);
  const req = REQUIRED[m.kind];
  if (!req) throw new Refused("contract", "unknown kind");
  for (const k of req) if (!(k in m)) throw new Refused("contract", `missing ${k}`);
  if (typeof m.robot !== "string" || !ROBOT_RE.test(m.robot)) throw new Refused("contract", "robot");
  if (typeof m.frame !== "string" || !FRAME_RE.test(m.frame)) throw new Refused("contract", "frame");
  if (!num(m.stamp)) throw new Refused("contract", "stamp");
  if (m.kind === "MapSnapshot") {
    if (!int(m.seq, 0)) throw new Refused("contract", "seq");
    checkGridHeader(m, "MapSnapshot");
  } else if (m.kind === "PoseSample") {
    if (![m.x, m.y, m.yaw_deg].every(num)) throw new Refused("contract", "pose");
  } else if (m.kind === "Overlays") {
    if (!Array.isArray(m.keepouts) || !Array.isArray(m.zones)) throw new Refused("contract", "overlays");
    m.keepouts.forEach((k, i) => checkGridHeader(k && k.grid, `keepouts[${i}]`));
    for (const z of m.zones)
      if (!z || !Array.isArray(z.polygon) || z.polygon.length < 3 ||
          !z.polygon.every((p) => Array.isArray(p) && p.length === 2 && p.every(num)))
        throw new Refused("contract", "zone");
    if (m.target !== null && (!m.target || !num(m.target.x) || !num(m.target.y)))
      throw new Refused("contract", "target");
  } else if (m.kind === "SessionStatus") {
    if (!["started", "running", "stopped"].includes(m.phase)) throw new Refused("contract", "phase");
    if (typeof m.text !== "string" || m.text.length > MAX_TEXT) throw new Refused("contract", "text");
  }
}

function b64bytes(s) {
  let bin;
  try { bin = atob(s); } catch (e) { throw new Refused("contract", "cells is not base64"); }
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

async function inflate(bytes) {
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

const canDigest = !!(globalThis.crypto && crypto.subtle);
async function sha256hex(bytes) {
  const d = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return Array.from(d, (b) => b.toString(16).padStart(2, "0")).join("");
}

// A grid's cells back to raw values (occupancy + 1), row-major from row 0, checked.
async function decodeGrid(f) {
  const png = b64bytes(f.cells);
  const sig = [137, 80, 78, 71, 13, 10, 26, 10];
  if (!sig.every((b, i) => png[i] === b)) throw new Refused("contract", "cells is not a PNG");
  const dv = new DataView(png.buffer);
  let pos = 8, w = null, h = null;
  const idat = [];
  while (pos + 8 <= png.length) {
    const n = dv.getUint32(pos);
    const kind = String.fromCharCode(...png.subarray(pos + 4, pos + 8));
    const data = png.subarray(pos + 8, pos + 8 + n);
    pos += 12 + n;
    if (kind === "IHDR") {
      w = dv.getUint32(pos - 12 - n + 8); h = dv.getUint32(pos - 12 - n + 12);
      if (data[8] !== 8 || data[9] !== 0 || data[12] !== 0)
        throw new Refused("contract", "cells must be 8-bit grayscale");
    } else if (kind === "IDAT") idat.push(data);
    else if (kind === "IEND") break;
  }
  if (w !== f.width || h !== f.height) throw new Refused("contract", "PNG size");
  let rows;
  try { rows = await inflate(new Uint8Array(await new Blob(idat).arrayBuffer())); }
  catch (e) { throw new Refused("contract", "PNG data does not inflate"); }
  if (rows.length !== h * (w + 1)) throw new Refused("contract", "PNG data length");
  const raw = new Uint8Array(w * h);
  for (let r = 0; r < h; r++) {
    if (rows[r * (w + 1)] !== 0) throw new Refused("contract", "PNG filter");
    raw.set(rows.subarray(r * (w + 1) + 1, (r + 1) * (w + 1)), r * w);
  }
  if (canDigest) {
    if ((await sha256hex(raw)) !== f.sha256) throw new Refused("contract", "sha256");
  } else view.unchecked++;
  for (const v of raw) if (v > 101) throw new Refused("contract", "cell value");
  return { w, h, raw };
}

function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function rgb(hex) {
  const v = parseInt(hex.replace("#", ""), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

// A grid drawn once into a canvas of its own size, row 0 at the bottom.
function paint(grid, colour) {
  const cv = document.createElement("canvas");
  cv.width = grid.w; cv.height = grid.h;
  const ctx = cv.getContext("2d");
  const img = ctx.createImageData(grid.w, grid.h);
  for (let r = 0; r < grid.h; r++) {
    const y = grid.h - 1 - r;
    for (let x = 0; x < grid.w; x++) {
      const c = colour(grid.raw[r * grid.w + x]);
      const i = (y * grid.w + x) * 4;
      img.data[i] = c[0]; img.data[i + 1] = c[1]; img.data[i + 2] = c[2]; img.data[i + 3] = c[3];
    }
  }
  ctx.putImageData(img, 0, 0);
  return cv;
}

function mapColour() {
  const free = rgb(css("--free")), occ = rgb(css("--occ")), unk = rgb(css("--unknown"));
  return (v) => {
    if (v === 0) return [...unk, 255];
    const p = (v - 1) / 100;                      // occupancy 0..1
    if (p >= 0.65) return [...occ, 255];
    return [0, 1, 2].map((i) => Math.round(free[i] + (unk[i] - free[i]) * Math.min(1, p / 0.65)))
      .concat(255);
  };
}

function keepColour() {
  const k = rgb(css("--keep"));
  return (v) => (v === 101 ? [...k, 110] : [0, 0, 0, 0]);
}

// -- what the viewer holds -----------------------------------------------------------------------

const view = {
  robot: null, frame: null, map: null, mapCanvas: null, mapSha: null, mapChangedAt: null,
  poses: [], lastPose: null, overlays: { keepouts: [], zones: [], target: null }, keepCanvases: [],
  status: null, arrived: {}, refused: { version: 0, contract: 0, frame: 0 },
  versionsSeen: new Set(), accepted: 0, unchecked: 0,
};

function resetSession(frame) {
  Object.assign(view, { frame, map: null, mapCanvas: null, mapSha: null, mapChangedAt: null,
    poses: [], lastPose: null, overlays: { keepouts: [], zones: [], target: null },
    keepCanvases: [], status: null, arrived: {} });
}

const now = () => performance.now() / 1000;

async function accept(m) {
  try {
    check(m);
    if (view.frame === null) view.frame = m.frame;
    else if (m.frame !== view.frame) {
      if (m.kind === "SessionStatus" && m.phase === "started") resetSession(m.frame);
      else throw new Refused("frame", m.frame);
    }
    if (m.kind === "MapSnapshot") {
      const g = await decodeGrid(m);
      if (!view.map || m.seq >= view.map.seq) {
        if (m.sha256 !== view.mapSha) {
          view.mapChangedAt = now();
          view.mapCanvas = paint(g, mapColour());
        }
        view.map = m; view.mapSha = m.sha256;
      }
    } else if (m.kind === "PoseSample") {
      view.lastPose = m;
      const last = view.poses[view.poses.length - 1];
      if (!last || Math.hypot(m.x - last.x, m.y - last.y) >= PATH_STEP_M) view.poses.push(m);
      if (view.poses.length > 20000) view.poses.splice(0, view.poses.length - 20000);
    } else if (m.kind === "Overlays") {
      const canvases = [];
      for (const k of m.keepouts) canvases.push({ g: k.grid, cv: paint(await decodeGrid(k.grid), keepColour()) });
      view.overlays = m; view.keepCanvases = canvases;
    } else if (m.kind === "SessionStatus") {
      view.status = m;
    }
    view.arrived[m.kind] = now();
    view.accepted++;
  } catch (e) {
    const why = e instanceof Refused ? e.why : "contract";
    view.refused[why]++;
    if (why === "version") view.versionsSeen.add(e.message);
  }
}

// Messages are taken one at a time, in arrival order: a grid's decode is asynchronous.
let chain = Promise.resolve();
function take(text) {
  let m = null;
  try { m = JSON.parse(text); } catch (e) { m = null; }
  chain = chain.then(() => accept(m));
}

// -- drawing ---------------------------------------------------------------------------------------

function staleness(t) {
  if (view.status && view.status.phase === "stopped") return { pose: null, map: null, ended: true };
  const out = { pose: null, map: null, ended: false };
  if ("PoseSample" in view.arrived) {
    const age = t - view.arrived.PoseSample;
    out.pose = age > POSE_LOST_S ? "lost" : age > POSE_STALE_S ? "stale" : "fresh";
  }
  if ("MapSnapshot" in view.arrived) out.map = t - view.arrived.MapSnapshot > MAP_STALE_S ? "stale" : "fresh";
  return out;
}

function describeSource(source, simulated) {
  let text;
  if (source.startsWith("guided:")) text = `Guided session ${source.slice(7)}: an operator drives the robot.`;
  else if (source.startsWith("replay:")) text = `Replay of ${source.slice(7)}.`;
  else text = `${source}: the robot explores inside its bound.`;
  return text + (simulated ? " Simulated: no real robot." : "");
}

function draw() {
  const cv = document.getElementById("map");
  const dpr = window.devicePixelRatio || 1;
  const W = cv.clientWidth, H = cv.clientHeight;
  if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) {
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
  }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const t = now(), fresh = staleness(t);
  // The world box: the map's extent, or a box round the robot before any map.
  let x0 = -4, y0 = -4, x1 = 4, y1 = 4;
  if (view.map) {
    x0 = view.map.origin[0]; y0 = view.map.origin[1];
    x1 = x0 + view.map.width * view.map.resolution; y1 = y0 + view.map.height * view.map.resolution;
  } else if (view.lastPose) {
    x0 = view.lastPose.x - 4; x1 = view.lastPose.x + 4; y0 = view.lastPose.y - 4; y1 = view.lastPose.y + 4;
  }
  const pad = 12, s = Math.min((W - 2 * pad) / (x1 - x0), (H - 2 * pad) / (y1 - y0));
  const ox = (W - s * (x1 - x0)) / 2, oy = (H - s * (y1 - y0)) / 2;
  const X = (x) => ox + (x - x0) * s, Y = (y) => H - oy - (y - y0) * s;
  ctx.imageSmoothingEnabled = false;
  const grid = (g, canvas) => ctx.drawImage(canvas, X(g.origin[0]), Y(g.origin[1] + g.height * g.resolution),
    g.width * g.resolution * s, g.height * g.resolution * s);
  if (view.map && view.mapCanvas) {
    ctx.globalAlpha = fresh.map === "stale" ? 0.45 : 1;
    grid(view.map, view.mapCanvas);
    ctx.globalAlpha = 1;
  }
  for (const k of view.keepCanvases) grid(k.g, k.cv);
  ctx.lineWidth = 2;
  ctx.strokeStyle = css("--zone");
  for (const z of view.overlays.zones) {
    ctx.beginPath();
    z.polygon.forEach(([x, y], i) => (i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))));
    ctx.closePath(); ctx.stroke();
  }
  if (view.poses.length > 1) {
    ctx.strokeStyle = css("--path"); ctx.lineWidth = 2.5; ctx.beginPath();
    view.poses.forEach((p, i) => (i ? ctx.lineTo(X(p.x), Y(p.y)) : ctx.moveTo(X(p.x), Y(p.y))));
    ctx.stroke();
  }
  const tg = view.overlays.target;
  if (tg) {
    ctx.strokeStyle = css("--target"); ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(X(tg.x), Y(tg.y), 11, 0, 2 * Math.PI); ctx.stroke();
    ctx.fillStyle = css("--ink"); ctx.font = "600 13px system-ui, sans-serif";
    ctx.fillText(tg.label, X(tg.x) + 14, Y(tg.y) - 10);
  }
  const p = view.lastPose;
  if (p) {
    const live = fresh.pose === "fresh" || fresh.pose === null;
    const colour = live ? css("--robot") : css("--stale");
    const a = (p.yaw_deg * Math.PI) / 180;
    if (fresh.pose !== "lost") {
      ctx.strokeStyle = colour; ctx.lineWidth = 3; ctx.beginPath();
      ctx.moveTo(X(p.x), Y(p.y));
      ctx.lineTo(X(p.x) + 22 * Math.cos(a), Y(p.y) - 22 * Math.sin(a));
      ctx.stroke();
    }
    ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), 7, 0, 2 * Math.PI);
    ctx.lineWidth = 2.5; ctx.strokeStyle = colour; ctx.stroke();
    if (live) { ctx.fillStyle = colour; ctx.fill(); }
  }
  panel(t, fresh);
}

function setText(id, text, stale) {
  const el = document.getElementById(id);
  el.textContent = text;
  el.classList.toggle("stale", !!stale);
}

function panel(t, fresh) {
  const st = view.status;
  setText("source", st ? describeSource(st.source, !!st.simulated) : "No session yet.");
  document.getElementById("sim").hidden = !(st && st.simulated);
  setText("phase", st ? (fresh.ended ? "stopped: the session ended" : st.phase) : "-");
  setText("text", st && st.text ? st.text : "-");
  const p = view.lastPose;
  if (!p) setText("pose", "No pose yet.");
  else {
    let text = `(${p.x.toFixed(2)}, ${p.y.toFixed(2)}) m, heading ${Math.round(p.yaw_deg)}°`;
    const age = t - view.arrived.PoseSample;
    if (fresh.pose === "stale") text += `. Pose ${age.toFixed(0)} s old.`;
    if (fresh.pose === "lost") text += `. Pose lost: ${age.toFixed(0)} s old.`;
    setText("pose", text, fresh.pose === "stale" || fresh.pose === "lost");
  }
  if (!view.map) setText("mapinfo", "No map yet.");
  else {
    let text = `#${view.map.seq}, ${view.map.width} x ${view.map.height} cells at ${view.map.resolution} m, ` +
      `changed ${(t - view.mapChangedAt).toFixed(0)} s ago`;
    if (fresh.map === "stale") text += `. No snapshot for ${(t - view.arrived.MapSnapshot).toFixed(0)} s.`;
    setText("mapinfo", text, fresh.map === "stale");
  }
  setText("area", st && st.area_free_m2 !== null ? `${st.area_free_m2.toFixed(1)} m²` : "-");
  setText("frame", view.frame || "-");
  const r = view.refused, n = r.version + r.contract + r.frame;
  let refused = String(n);
  if (n) {
    refused += ` (version ${r.version}${view.versionsSeen.size ? ": " + [...view.versionsSeen].join(", ") : ""}` +
      `; contract ${r.contract}; frame ${r.frame})`;
  }
  if (view.unchecked) refused += `. ${view.unchecked} grids not checked: this browser has no SHA-256 here.`;
  setText("refused", refused, n > 0);
  const b = document.body.dataset;
  Object.assign(b, { accepted: view.accepted, refusedVersion: r.version, refusedContract: r.contract,
    refusedFrame: r.frame, frame: view.frame || "", mapSeq: view.map ? view.map.seq : "",
    poses: view.poses.length, phase: st ? st.phase : "", keepouts: view.keepCanvases.length,
    digest: canDigest ? "checked" : "unchecked", pose: fresh.pose || "", source: st ? st.source : "" });
}

// -- the stream ------------------------------------------------------------------------------------

let source = null;
function connect(robot) {
  if (source) source.close();
  view.robot = robot;
  resetSession(null);
  view.refused = { version: 0, contract: 0, frame: 0 }; view.versionsSeen = new Set(); view.accepted = 0;
  const link = document.getElementById("link");
  source = new EventSource(`v0/events?robot=${encodeURIComponent(robot)}`);
  source.onopen = () => { link.textContent = "Connected"; link.classList.remove("warn"); };
  source.onerror = () => { link.textContent = "Reconnecting"; link.classList.add("warn"); };
  source.onmessage = (e) => take(e.data);
}

async function robots() {
  const sel = document.getElementById("robot");
  try {
    const r = await (await fetch("v0/robots", { cache: "no-store" })).json();
    const names = r.robots.map((x) => x.robot);
    const want = new URLSearchParams(location.search).get("robot");
    for (const n of names) {
      if (![...sel.options].some((o) => o.value === n)) sel.add(new Option(n, n));
    }
    if (!view.robot) {
      const first = want && ROBOT_RE.test(want) ? want : names[0];
      if (first) {
        if (![...sel.options].some((o) => o.value === first)) sel.add(new Option(first, first));
        sel.value = first; connect(first);
      }
    }
  } catch (e) { /* the service is not answering yet; asked again below */ }
}

document.getElementById("robot").addEventListener("change", (e) => connect(e.target.value));
robots();
setInterval(robots, 5000);
setInterval(draw, 250);
window.addEventListener("resize", draw);
