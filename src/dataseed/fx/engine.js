/* DataSeed FX engine v1.0.0 - purely decorative, cursor-reactive data-particle background.
 *
 * Injected ONCE into the page by dataseed.visual_theme. Design rules:
 *  - one <canvas id="dataseed-fx-canvas"> at z-index:-1, pointer-events:none (behind all app content);
 *  - one requestAnimationFrame loop, one set of passive document listeners (never preventDefault /
 *    stopPropagation), a singleton on window.__dataseedFx so Streamlit reruns cannot duplicate anything;
 *  - on any doubt (odd stacking context, repeated errors) it removes itself and the app is left untouched;
 *  - <html data-ds-fx="on"> is set only while the canvas is mounted; the theme CSS only makes app
 *    backgrounds transparent under that attribute.
 */
(function (global) {
  "use strict";
  var VERSION = "1.0.0";
  if (!global || !global.document) { return; }
  var doc = global.document;
  if (global.__dataseedFx) {
    if (global.__dataseedFx.version === VERSION) { return; }
    try { global.__dataseedFx.destroy(); } catch (e) { /* ignore */ }
  }

  var CANVAS_ID = "dataseed-fx-canvas";
  var PULSE_MS = 850, PULSE_MS_CALM = 600, MAX_PULSES = 6;
  var PALETTE = [[110, 255, 200], [70, 235, 255], [120, 255, 150], [150, 225, 255]];
  var GLYPHS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz".split("")
    .concat(["\u03A3", "\u03C0", "\u03BB", "\u221E", "\u2211", "\u221A", "\u0394", "{", "}", "[", "]", "<", ">", "/", "%", "+", "=", "*", "!", "?"]);
  var WORDS = ["42", "108", "data", "D a t a", "0101", "1101", "7", "9", "{ }", "</>", "x\u00B2", "\u03A3x", "if", "NaN",
    "\u03BC", "\u03C3", "101", "0x7F", "1e3", "\u2202", "\u2248", "seed", "rows", "x q", "a t a"];
  var SIZES = [11, 13, 16, 20];

  var canvas = null, ctx = null, sprite = null;
  var w = 0, h = 0, dpr = 1;
  var particles = [], byLevel = [[], [], [], []], nodes = [];
  var pulses = [];
  var pointer = { x: -9999, y: -9999, active: false };
  var mode = "off", mounted = false, running = false;
  var raf = 0, last = 0, errors = 0, resizeTimer = 0;
  var mq = null, costEma = 0, costFrames = 0, shrinks = 0, frameCount = 0;
  var listeners = [];

  function now() { return global.performance && global.performance.now ? global.performance.now() : Date.now(); }
  function clamp(v, a, b) { return v < a ? a : (v > b ? b : v); }
  function calm() { return mode === "calm" || !!(mq && mq.matches); }

  // ---------------------------------------------------------------- particles
  function makeParticle(spread) {
    var z = 0.35 + Math.random() * 0.65;
    var ang = Math.random() * Math.PI * 2;
    var sp = (5 + Math.random() * 18) * z;
    var dot = Math.random() < 0.14;
    var col = PALETTE[(Math.random() * PALETTE.length) | 0];
    var p = {
      x: Math.random() * w, y: spread ? Math.random() * h : Math.random() * h,
      vx: Math.cos(ang) * sp, vy: Math.sin(ang) * sp,
      ox: 0, oy: 0, ovx: 0, ovy: 0,
      z: z, level: Math.min(3, Math.floor((z - 0.35) / 0.65 * 4)),
      glyph: dot ? "" : (Math.random() < 0.28 ? WORDS[(Math.random() * WORDS.length) | 0] : GLYPHS[(Math.random() * GLYPHS.length) | 0]),
      dot: dot, rgb: col[0] + "," + col[1] + "," + col[2],
      a: (0.07 + Math.random() * 0.26) * (0.45 + 0.55 * z),
      ph: Math.random() * 6.283, ts: 0.4 + Math.random() * 1.4,
      node: false, g: 0
    };
    p.node = !dot && Math.random() < 0.34;
    return p;
  }

  function targetCount() {
    var n = clamp(Math.round(w * h / 16000), 45, 120);
    if (calm()) { n = Math.round(n * 0.55); }
    return Math.max(30, Math.round(n * Math.pow(0.75, shrinks)));
  }

  function rebuildGroups() {
    byLevel = [[], [], [], []]; nodes = [];
    for (var i = 0; i < particles.length; i++) {
      byLevel[particles[i].level].push(particles[i]);
      if (particles[i].node) { nodes.push(particles[i]); }
    }
  }

  function fillParticles() {
    var n = targetCount();
    while (particles.length < n) { particles.push(makeParticle(true)); }
    if (particles.length > n) { particles.length = n; }
    rebuildGroups();
  }

  function makeSprite() {
    var s = doc.createElement("canvas");
    s.width = 64; s.height = 64;
    var c = s.getContext && s.getContext("2d");
    if (!c || !c.createRadialGradient) { return null; }
    var g = c.createRadialGradient(32, 32, 0, 32, 32, 32);
    g.addColorStop(0, "rgba(140,255,225,0.55)");
    g.addColorStop(0.35, "rgba(70,235,255,0.18)");
    g.addColorStop(1, "rgba(70,235,255,0)");
    c.fillStyle = g; c.fillRect(0, 0, 64, 64);
    return s;
  }

  // ---------------------------------------------------------------- pulses
  function addPulse(x, y) {
    if (mode === "off" || !mounted) { return; }
    var maxR = Math.sqrt(Math.max(x, w - x) * Math.max(x, w - x) + Math.max(y, h - y) * Math.max(y, h - y));
    pulses.push({ t0: now(), x: x, y: y, maxR: maxR });
    if (pulses.length > MAX_PULSES) { pulses.shift(); }
  }

  function pulseMs() { return calm() ? PULSE_MS_CALM : PULSE_MS; }

  /** Glow (0..1.4) at a point: a brief global lift plus an expanding ripple from each recent click. */
  function glowAt(x, y, t) {
    var g = 0, dur = pulseMs(), c = calm();
    for (var i = 0; i < pulses.length; i++) {
      var pu = pulses[i], u = (t - pu.t0) / dur;
      if (u < 0 || u >= 1) { continue; }
      var env = u < 0.14 ? u / 0.14 : Math.pow(1 - (u - 0.14) / 0.86, 2);
      g += env * (c ? 0.25 : 0.38);
      if (!c) {
        var r = (1 - Math.pow(1 - u, 3)) * pu.maxR;
        var d = Math.sqrt((x - pu.x) * (x - pu.x) + (y - pu.y) * (y - pu.y));
        var q = (d - r) / 110;
        g += Math.exp(-q * q) * Math.pow(1 - u, 1.2);
      }
    }
    return g > 1.4 ? 1.4 : g;
  }

  function expirePulses(t) {
    var dur = pulseMs(), keep = [];
    for (var i = 0; i < pulses.length; i++) { if (t - pulses[i].t0 < dur) { keep.push(pulses[i]); } }
    pulses = keep;
  }

  // ---------------------------------------------------------------- frame
  function update(dt, t) {
    var c = calm(), speed = c ? 0.35 : 1, R = 150, K = 7, C = 5, push = 950;
    expirePulses(t);
    for (var i = 0; i < particles.length; i++) {
      var p = particles[i];
      p.x += p.vx * dt * speed; p.y += p.vy * dt * speed;
      if (p.x < -40) { p.x = w + 30; } else if (p.x > w + 40) { p.x = -30; }
      if (p.y < -40) { p.y = h + 30; } else if (p.y > h + 40) { p.y = -30; }
      var ax = -K * p.ox - C * p.ovx, ay = -K * p.oy - C * p.ovy;
      if (pointer.active && !c) {
        var dx = p.x + p.ox - pointer.x, dy = p.y + p.oy - pointer.y, d2 = dx * dx + dy * dy;
        if (d2 < R * R && d2 > 0.01) {
          var d = Math.sqrt(d2), f = Math.pow(1 - d / R, 2) * push * (0.5 + 0.5 * p.z);
          var nx = dx / d, ny = dy / d;
          ax += nx * f + (-ny) * f * 0.22;     // push away, plus a slight swirl so particles flow around the cursor
          ay += ny * f + nx * f * 0.22;
        }
      }
      p.ovx += ax * dt; p.ovy += ay * dt; p.ox += p.ovx * dt; p.oy += p.ovy * dt;
      var m = Math.sqrt(p.ox * p.ox + p.oy * p.oy);
      if (m > 95) { p.ox *= 95 / m; p.oy *= 95 / m; }
      p.g = pulses.length ? glowAt(p.x + p.ox, p.y + p.oy, t) : 0;
    }
  }

  function draw(t) {
    ctx.clearRect(0, 0, w, h);
    var sec = t / 1000, c = calm(), i, j, p;
    // network lines between a small subset of nearby "node" particles
    var L = w < 700 ? 90 : 135, L2 = L * L, drawn = 0;
    ctx.lineWidth = 1;
    for (i = 0; i < nodes.length && drawn < 140; i++) {
      var a = nodes[i];
      for (j = i + 1; j < nodes.length && drawn < 140; j++) {
        var b = nodes[j], dx = (a.x + a.ox) - (b.x + b.ox), dy = (a.y + a.oy) - (b.y + b.oy), d2 = dx * dx + dy * dy;
        if (d2 < L2) {
          var al = (1 - Math.sqrt(d2) / L) * 0.2 * (0.5 + 0.5 * Math.min(a.z, b.z)) + (a.g + b.g) * 0.09;
          ctx.strokeStyle = "rgba(90,240,225," + clamp(al, 0, 0.7).toFixed(3) + ")";
          ctx.beginPath(); ctx.moveTo(a.x + a.ox, a.y + a.oy); ctx.lineTo(b.x + b.ox, b.y + b.oy); ctx.stroke();
          drawn++;
        }
      }
    }
    // glyphs, grouped by font size so the font is set only four times per frame
    ctx.textBaseline = "middle"; ctx.textAlign = "center";
    for (var lv = 0; lv < 4; lv++) {
      ctx.font = SIZES[lv] + "px ui-monospace, 'JetBrains Mono', 'Fira Code', Menlo, Consolas, monospace";
      var grp = byLevel[lv];
      for (i = 0; i < grp.length; i++) {
        p = grp[i];
        var tw = Math.pow(0.5 + 0.5 * Math.sin(sec * p.ts + p.ph), 6);       // occasional soft flare
        var al2 = clamp(p.a * (0.7 + 0.9 * tw) + p.g * 0.62, 0, 1);
        var x = p.x + p.ox, y = p.y + p.oy;
        if (p.g > 0.1 && sprite) {
          var r = (10 + 18 * p.z) * (1 + 0.5 * p.g);
          ctx.globalCompositeOperation = "lighter"; ctx.globalAlpha = clamp(p.g * 0.55, 0, 0.8);
          ctx.drawImage(sprite, x - r, y - r, r * 2, r * 2);
          ctx.globalCompositeOperation = "source-over"; ctx.globalAlpha = 1;
        }
        ctx.fillStyle = "rgba(" + p.rgb + "," + al2.toFixed(3) + ")";
        if (p.dot) {
          ctx.beginPath(); ctx.arc(x, y, (1 + 1.4 * p.z) * (1 + 0.35 * p.g), 0, 6.283); ctx.fill();
        } else if (p.g > 0.1) {
          var s = 1 + 0.22 * Math.min(p.g, 1);
          ctx.save(); ctx.translate(x, y); ctx.scale(s, s); ctx.fillText(p.glyph, 0, 0); ctx.restore();
        } else {
          ctx.fillText(p.glyph, x, y);
        }
      }
    }
    // click ripple rings (full mode only)
    if (!c) {
      var dur = PULSE_MS;
      for (i = 0; i < pulses.length; i++) {
        var pu = pulses[i], u = (t - pu.t0) / dur;
        if (u < 0 || u >= 1) { continue; }
        var rr = (1 - Math.pow(1 - u, 3)) * pu.maxR, fade = Math.pow(1 - u, 1.6);
        ctx.strokeStyle = "rgba(110,255,220," + (0.3 * fade).toFixed(3) + ")"; ctx.lineWidth = 1.4;
        ctx.beginPath(); ctx.arc(pu.x, pu.y, rr, 0, 6.283); ctx.stroke();
        ctx.strokeStyle = "rgba(70,235,255," + (0.14 * fade).toFixed(3) + ")"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(pu.x, pu.y, rr * 0.82, 0, 6.283); ctx.stroke();
      }
    }
  }

  function frame() {
    raf = 0;
    if (!running) { return; }
    raf = global.requestAnimationFrame(frame);
    var t = now(), dt = clamp((t - last) / 1000, 0.001, 0.05);
    last = t;
    try {
      update(dt, t); draw(t); frameCount++; errors = 0;
    } catch (e) {
      if (++errors >= 3) { fail(e); }
      return;
    }
    var cost = now() - t;                       // adaptive quality: shed particles on slow machines
    costEma = costEma ? costEma * 0.95 + cost * 0.05 : cost;
    if (++costFrames >= 90) {
      if (costEma > 9 && shrinks < 2) { shrinks++; fillParticles(); }
      costFrames = 0;
    }
  }

  function fail(e) {
    try { if (global.console && global.console.warn) { global.console.warn("[DataSeed FX] disabled after errors:", e); } } catch (x) { /* ignore */ }
    stop(); unmount(); mode = "off";
  }

  // ---------------------------------------------------------------- lifecycle
  function resize() {
    if (!canvas) { return; }
    var nw = global.innerWidth || 1280, nh = global.innerHeight || 720;
    var sx = w ? nw / w : 1, sy = h ? nh / h : 1;
    w = nw; h = nh; dpr = Math.min(global.devicePixelRatio || 1, 1.5);
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    for (var i = 0; i < particles.length; i++) { particles[i].x *= sx; particles[i].y *= sy; }
    fillParticles();
  }

  function onResize() {
    if (resizeTimer) { global.clearTimeout(resizeTimer); }
    resizeTimer = global.setTimeout(function () { resizeTimer = 0; resize(); }, 120);
  }
  function onMove(e) { if (e.pointerType === "touch") { return; } pointer.x = e.clientX; pointer.y = e.clientY; pointer.active = true; }
  function onOut(e) { if (!e.relatedTarget) { pointer.active = false; } }
  function onDown(e) { addPulse(e.clientX, e.clientY); }
  function onClick(e) {                       // keyboard-activated clicks have detail === 0 and no pointerdown
    if (e.detail !== 0) { return; }
    var x = w / 2, y = h / 2;
    if (e.target && e.target.getBoundingClientRect) {
      var r = e.target.getBoundingClientRect(); x = r.left + r.width / 2; y = r.top + r.height / 2;
    }
    addPulse(x, y);
  }
  function onVisibility() { if (doc.hidden) { stopLoop(); } else if (mounted && mode !== "off") { startLoop(); } }
  function onMQ() { fillParticles(); }

  function listen(target, type, fn, opts) {
    target.addEventListener(type, fn, opts);
    listeners.push([target, type, fn, opts]);
  }

  function stackingContext(el) {
    var cs = global.getComputedStyle ? global.getComputedStyle(el) : null;
    if (!cs) { return false; }
    return (cs.position !== "static" && cs.zIndex !== "auto") || (cs.transform && cs.transform !== "none") ||
      (cs.filter && cs.filter !== "none") || cs.isolation === "isolate" || parseFloat(cs.opacity) < 1 ||
      (cs.perspective && cs.perspective !== "none");
  }

  function mount() {
    if (mounted) { return true; }
    try {
      if (!doc.body || stackingContext(doc.body)) {
        if (global.console && global.console.warn) { global.console.warn("[DataSeed FX] page layout not compatible; background disabled."); }
        return false;
      }
      var old = doc.getElementById(CANVAS_ID);
      if (old && old.parentNode) { old.parentNode.removeChild(old); }
      canvas = doc.createElement("canvas");
      canvas.id = CANVAS_ID;
      canvas.setAttribute("aria-hidden", "true"); canvas.setAttribute("role", "presentation");
      canvas.style.cssText = "position:fixed;left:0;top:0;width:100%;height:100%;z-index:-1;pointer-events:none;display:block;";
      ctx = canvas.getContext("2d");
      if (!ctx) { canvas = null; return false; }
      doc.body.insertBefore(canvas, doc.body.firstChild);
      sprite = makeSprite();
      w = 0; h = 0; particles = []; pulses = []; shrinks = 0;
      resize();
      var o = { capture: true, passive: true };
      listen(doc, "pointermove", onMove, o); listen(doc, "pointerout", onOut, o);
      listen(doc, "pointerdown", onDown, o); listen(doc, "click", onClick, o);
      listen(doc, "visibilitychange", onVisibility, false); listen(global, "resize", onResize, false);
      if (global.matchMedia) {
        mq = global.matchMedia("(prefers-reduced-motion: reduce)");
        if (mq.addEventListener) { listen(mq, "change", onMQ, false); }
      }
      doc.documentElement.setAttribute("data-ds-fx", "on");
      mounted = true;
      return true;
    } catch (e) {
      unmount();
      return false;
    }
  }

  function unmount() {
    stopLoop();
    for (var i = 0; i < listeners.length; i++) {
      var l = listeners[i];
      try { l[0].removeEventListener(l[1], l[2], l[3]); } catch (e) { /* ignore */ }
    }
    listeners = [];
    if (resizeTimer) { global.clearTimeout(resizeTimer); resizeTimer = 0; }
    if (canvas && canvas.parentNode) { canvas.parentNode.removeChild(canvas); }
    canvas = null; ctx = null; sprite = null; particles = []; nodes = []; pulses = []; byLevel = [[], [], [], []];
    try { doc.documentElement.removeAttribute("data-ds-fx"); } catch (e) { /* ignore */ }
    mounted = false;
  }

  function startLoop() { if (running || !mounted || doc.hidden) { return; } running = true; last = now(); raf = global.requestAnimationFrame(frame); }
  function stopLoop() { running = false; if (raf) { global.cancelAnimationFrame(raf); raf = 0; } }
  function stop() { stopLoop(); }

  function configure(cfg) {
    var m = String((cfg && cfg.mode) || "full").toLowerCase();
    if (m !== "off" && m !== "calm") { m = "full"; }
    if (m === mode && (m === "off" || (mounted && running))) { return; }   // same settings: nothing to do
    mode = m;
    if (mode === "off") { unmount(); return; }
    if (!mounted && !mount()) { mode = "off"; return; }
    fillParticles();
    startLoop();
  }

  function destroy() { mode = "off"; unmount(); if (global.__dataseedFx === api) { global.__dataseedFx = null; } }

  var api = {
    version: VERSION, configure: configure, destroy: destroy, pulse: addPulse,
    stats: function () {
      return { mode: mode, mounted: mounted, running: running, particles: particles.length, pulses: pulses.length,
               listeners: listeners.length, frames: frameCount, costMs: costEma };
    }
  };
  global.__dataseedFx = api;
})(typeof window !== "undefined" ? window : this);
