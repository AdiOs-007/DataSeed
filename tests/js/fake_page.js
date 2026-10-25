/* Runs src/dataseed/fx/engine.js against a fake DOM with a manual clock. Prints "OK <n> checks" or exits 1.
 * This verifies the engine's LOGIC (singleton, timing, listeners, cleanup, fallbacks). It is not a real browser. */
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const CODE = fs.readFileSync(path.join(__dirname, "..", "..", "src", "dataseed", "fx", "engine.js"), "utf8");

function lcg(seed) { let s = seed >>> 0; return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296); }

function makeEnv(opts) {
  opts = opts || {};
  const log = { fillText: [], drawImage: 0, arc: 0, fillTextAlpha: [], timersSet: 0, timersCleared: 0, rafSet: 0, rafCancelled: 0 };
  let clock = 1000, rafCb = null, rafId = 0, timers = [];
  const regs = [];                                   // all addEventListener registrations
  const attrs = {};
  const ctx = {
    clearRect() { log.fillText = []; log.drawImage = 0; log.arc = 0; log.fillTextAlpha = []; },
    fillText(g, x, y) { if (opts.throwOnDraw) throw new Error("boom"); log.fillText.push([x, y]); log.fillTextAlpha.push(parseFloat(ctx.fillStyle.split(",").pop())); },
    drawImage() { log.drawImage++; }, arc() { log.arc++; }, setTransform() {}, save() {}, restore() {}, translate() {}, scale() {},
    beginPath() {}, moveTo() {}, lineTo() {}, stroke() {}, fill() {}, fillRect() {}, createRadialGradient() { return { addColorStop() {} }; },
    fillStyle: "", strokeStyle: "", font: "", globalAlpha: 1, globalCompositeOperation: "source-over", lineWidth: 1,
  };
  function makeCanvas() { return { style: {}, attrs: {}, width: 0, height: 0, parentNode: null, setAttribute(k, v) { this.attrs[k] = v; }, getContext() { return ctx; } }; }
  const body = { children: [], insertBefore(el) { this.children.unshift(el); el.parentNode = this; }, removeChild(el) { this.children = this.children.filter(c => c !== el); el.parentNode = null; } };
  const doc = {
    hidden: false, body, documentElement: { setAttribute(k, v) { attrs[k] = v; }, removeAttribute(k) { delete attrs[k]; } },
    createElement(tag) {
      if (tag === "script") { return { attrs: {}, textContent: "", parentNode: null, setAttribute(k, v) { this.attrs[k] = v; } }; }
      return makeCanvas();
    },
    head: { appendChild(el) { el.parentNode = this; env.scriptEvals = (env.scriptEvals || 0) + 1; vm.runInContext(el.textContent, sandbox); }, removeChild(el) { el.parentNode = null; } },
    getElementById(id) { return body.children.find(c => c.id === id) || null; },
    addEventListener(t, f, o) { regs.push({ target: "doc", t, f, o }); },
    removeEventListener(t, f, o) { const i = regs.findIndex(r => r.target === "doc" && r.t === t && r.f === f); if (i >= 0) regs.splice(i, 1); },
  };
  const mqList = { matches: !!opts.reduceMotion, addEventListener(t, f) { regs.push({ target: "mq", t, f, o: false }); }, removeEventListener(t, f) { const i = regs.findIndex(r => r.target === "mq" && r.f === f); if (i >= 0) regs.splice(i, 1); } };
  const win = {
    document: doc, innerWidth: 1280, innerHeight: 720, devicePixelRatio: 2, console: { warn() { win._warned = (win._warned || 0) + 1; } },
    performance: { now: () => clock },
    requestAnimationFrame(cb) { log.rafSet++; rafCb = cb; return ++rafId; },
    cancelAnimationFrame() { log.rafCancelled++; rafCb = null; },
    setTimeout(fn, ms) { log.timersSet++; const t = { fn, id: timers.length + 1 }; timers.push(t); return t.id; },
    clearTimeout(id) { log.timersCleared++; timers = timers.filter(t => t.id !== id); },
    matchMedia() { return mqList; },
    getComputedStyle() { return opts.bodyStyle || { position: "static", zIndex: "auto", transform: "none", filter: "none", isolation: "auto", opacity: "1", perspective: "none" }; },
    addEventListener(t, f, o) { regs.push({ target: "win", t, f, o }); },
    removeEventListener(t, f) { const i = regs.findIndex(r => r.target === "win" && r.t === t && r.f === f); if (i >= 0) regs.splice(i, 1); },
    Math: Object.assign(Object.create(Math), { random: lcg(opts.seed || 7) }),
  };
  win.window = win;
  const sandbox = vm.createContext(win);
  vm.runInContext("var Math = window.Math;", sandbox);
  const env = {
    win, doc, ctx, log, regs, attrs, body,
    load() { vm.runInContext(CODE, sandbox); },
    fx() { return win.__dataseedFx; },
    advance(ms, step) { step = step || 16; for (let t = 0; t < ms; t += step) { clock += step; if (rafCb) { const cb = rafCb; rafCb = null; cb(clock); } } },
    now: () => clock, pendingTimers: () => timers.length, runTimers() { const t = timers; timers = []; t.forEach(x => x.fn()); },
    fire(target, type, ev) {
      const spies = { prevented: 0, stopped: 0 };
      const e = Object.assign({ preventDefault() { spies.prevented++; }, stopPropagation() { spies.stopped++; }, stopImmediatePropagation() { spies.stopped++; } }, ev || {});
      regs.filter(r => r.target === target && r.t === type).forEach(r => r.f(e));
      return spies;
    },
    hasRaf: () => rafCb !== null,
  };
  return env;
}


module.exports = { makeEnv, lcg, CODE };
