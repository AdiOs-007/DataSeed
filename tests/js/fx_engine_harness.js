/* Runs src/dataseed/fx/engine.js against a fake DOM with a manual clock. Prints "OK <n> checks" or exits 1.
 * This verifies the engine's LOGIC (singleton, timing, listeners, cleanup, fallbacks). It is not a real browser. */
"use strict";
const { makeEnv } = require("./fake_page");
let checks = 0;
function ok(cond, msg) { checks++; if (!cond) { console.error("FAIL: " + msg); process.exit(1); } }

// ---- 1. mount, singleton, passive listeners ------------------------------------------------
let E = makeEnv(); E.load();
ok(E.fx() && E.fx().version === "1.0.0", "engine registers window.__dataseedFx");
E.fx().configure({ mode: "full" });
ok(E.body.children.length === 1 && E.body.children[0].id === "dataseed-fx-canvas", "exactly one canvas inserted");
ok(E.body.children[0].style.cssText.includes("z-index:-1") && E.body.children[0].style.cssText.includes("pointer-events:none"), "canvas is behind content and non-interactive");
ok(E.body.children[0].attrs["aria-hidden"] === "true", "canvas hidden from assistive tech");
ok(E.attrs["data-ds-fx"] === "on", "html marker set while mounted");
const n0 = E.regs.length;
ok(n0 === 7, "7 listeners registered (got " + n0 + ")");
ok(E.regs.filter(r => ["pointermove", "pointerdown", "pointerout", "click"].includes(r.t)).every(r => r.o && r.o.passive === true && r.o.capture === true), "pointer/click listeners are passive+capture");
E.fx().configure({ mode: "full" }); E.fx().configure({ mode: "full" });
E.load(); E.load();                                     // simulates the loader re-running after Streamlit reruns
ok(E.body.children.length === 1 && E.regs.length === n0, "repeated configure/load never duplicates canvas or listeners");
ok(E.hasRaf() && E.log.rafSet === 1, "single animation loop");

// ---- 2. frames are cheap & bounded -----------------------------------------------------------
E.advance(16 * 60);
const st = E.fx().stats();
ok(st.frames === 60, "60 frames rendered");
ok(st.particles >= 45 && st.particles <= 120, "particle count within bounds: " + st.particles);
ok(E.log.fillText.length <= st.particles, "glyph draws per frame bounded");
ok(E.win._warned === undefined, "no warnings");

// ---- 3. click pulse: visible, < 1 s, fully faded ---------------------------------------------
E.advance(160);
ok(E.log.drawImage === 0, "no glow sprites before a click");
const sp0 = E.fire("doc", "pointerdown", { clientX: 400, clientY: 300 });
ok(sp0.prevented === 0 && sp0.stopped === 0, "pointerdown handler never blocks the event");
ok(E.fx().stats().pulses === 1, "click registers a pulse");
E.advance(112);                                         // ~ peak
ok(E.log.drawImage > 0, "background glows during the pulse (" + E.log.drawImage + " glowing particles)");
const peakGlow = E.log.drawImage;
E.advance(576);                                         // 688 ms after the click
ok(E.fx().stats().pulses === 1, "pulse still active at ~690 ms");
E.advance(192);                                         // 880 ms after the click
ok(E.fx().stats().pulses === 0, "pulse removed by 880 ms (< 1 s)");
E.advance(16);
ok(E.log.drawImage === 0, "glow completely gone after the pulse");

// ---- 4. rapid clicks ------------------------------------------------------------------------
const timersBefore = E.log.timersSet;
for (let i = 0; i < 200; i++) E.fire("doc", "pointerdown", { clientX: i * 5 % 1280, clientY: i * 3 % 720 });
ok(E.fx().stats().pulses <= 6, "pulse count capped at 6 (got " + E.fx().stats().pulses + ")");
ok(E.log.timersSet === timersBefore, "clicks create no timers");
E.advance(1000);
ok(E.fx().stats().pulses === 0 && E.log.rafSet >= 1 && E.fx().stats().running, "loop survives click storm and returns to idle");

// ---- 5. click semantics ---------------------------------------------------------------------
E.fire("doc", "pointerdown", { clientX: 10, clientY: 10 }); E.fire("doc", "click", { detail: 1, target: {} });
ok(E.fx().stats().pulses === 1, "mouse click = one pulse (pointerdown only)");
E.advance(1000);
E.fire("doc", "click", { detail: 0, target: { getBoundingClientRect: () => ({ left: 100, top: 50, width: 40, height: 20 }) } });
ok(E.fx().stats().pulses === 1, "keyboard-activated click also pulses");
const sp1 = E.fire("doc", "click", { detail: 0, target: {} });
ok(sp1.prevented === 0 && sp1.stopped === 0, "click handler never blocks the event");
E.advance(1000);

// ---- 6. resize debounced ---------------------------------------------------------------------
for (let i = 0; i < 30; i++) { E.win.innerWidth = 1000 + i; E.fire("win", "resize"); }
ok(E.pendingTimers() === 1, "30 resize events -> a single pending timer");
E.runTimers();
ok(E.body.children[0].width === Math.round(1029 * 1.5), "canvas resized with dpr capped at 1.5 (got " + E.body.children[0].width + ")");

// ---- 7. hidden tab pauses ---------------------------------------------------------------------
E.doc.hidden = true; E.fire("doc", "visibilitychange");
ok(!E.hasRaf() && !E.fx().stats().running, "animation pauses while tab hidden");
E.doc.hidden = false; E.fire("doc", "visibilitychange");
ok(E.hasRaf() && E.fx().stats().running, "animation resumes when visible");

// ---- 8. mode changes & destroy -----------------------------------------------------------------
E.fx().configure({ mode: "off" });
ok(E.body.children.length === 0 && E.regs.length === 0 && !E.attrs["data-ds-fx"] && !E.hasRaf(), "off: canvas, listeners, loop and marker all removed");
E.fx().configure({ mode: "full" });
ok(E.body.children.length === 1 && E.regs.length === n0, "re-enable restores exactly one canvas/listener set");
E.fx().destroy();
ok(E.win.__dataseedFx === null && E.body.children.length === 0 && E.regs.length === 0, "destroy cleans everything");

// ---- 9. cursor repels, then settles --------------------------------------------------------------
function run(withPointer) {
  const e = makeEnv({ seed: 11 }); e.load(); e.fx().configure({ mode: "full" });
  if (withPointer) { e.fire("doc", "pointermove", { clientX: 640, clientY: 360, pointerType: "mouse" }); }
  e.advance(16 * 40);
  const during = e.log.fillText.slice();
  if (withPointer) { e.fire("doc", "pointerout", { relatedTarget: null }); }
  e.advance(4000);
  return { during, after: e.log.fillText.slice() };
}
const A = run(true), B = run(false);
let maxDisp = 0, maxAfter = 0;
for (let i = 0; i < A.during.length; i++) maxDisp = Math.max(maxDisp, Math.hypot(A.during[i][0] - B.during[i][0], A.during[i][1] - B.during[i][1]));
for (let i = 0; i < A.after.length; i++) maxAfter = Math.max(maxAfter, Math.hypot(A.after[i][0] - B.after[i][0], A.after[i][1] - B.after[i][1]));
ok(A.during.length === B.during.length, "same particle count in both runs");
ok(maxDisp > 8 && maxDisp <= 100, "cursor displaces nearby particles smoothly (max " + maxDisp.toFixed(1) + " px)");
ok(maxAfter < 1.5, "particles settle back after the cursor leaves (residual " + maxAfter.toFixed(2) + " px)");

// ---- 10. calm mode / reduced motion ----------------------------------------------------------------
const C = makeEnv({ seed: 11 }); C.load(); C.fx().configure({ mode: "calm" });
const fullCount = A.during.length;
C.fire("doc", "pointermove", { clientX: 640, clientY: 360, pointerType: "mouse" }); C.advance(16 * 40);
const calmDuring = C.log.fillText.slice(); ok(calmDuring.length < fullCount, "calm mode uses fewer particles (" + calmDuring.length + " < " + fullCount + ")");
C.fire("doc", "pointerdown", { clientX: 300, clientY: 300 }); C.advance(16 * 20);
ok(C.log.arc <= 200 && C.fx().stats().pulses === 1, "calm pulse is a soft glow without ripple rings");
C.advance(700); ok(C.fx().stats().pulses === 0, "calm pulse ends within 600 ms");
const R = makeEnv({ reduceMotion: true }); R.load(); R.fx().configure({ mode: "full" }); R.advance(160);
ok(R.log.fillText.length < fullCount, "prefers-reduced-motion forces the calm experience");

// ---- 11. safety fallbacks ---------------------------------------------------------------------------------
const S = makeEnv({ bodyStyle: { position: "static", zIndex: "auto", transform: "translateZ(0)", filter: "none", isolation: "auto", opacity: "1", perspective: "none" } });
S.load(); S.fx().configure({ mode: "full" });
ok(S.body.children.length === 0 && !S.attrs["data-ds-fx"] && S.regs.length === 0, "incompatible stacking context: nothing mounted, page untouched");
const X = makeEnv({ throwOnDraw: true }); X.load(); X.fx().configure({ mode: "full" });
let threw = false; try { X.advance(16 * 10); } catch (e) { threw = true; }
ok(!threw && X.body.children.length === 0 && X.regs.length === 0 && !X.attrs["data-ds-fx"], "repeated draw errors: engine removes itself without throwing");
ok(X.win._warned === 1, "failure is logged once");
const D = makeEnv({}); D.load(); D.fx().configure({ mode: "bogus" });
ok(D.fx().stats().mode === "full", "unknown mode falls back to full");
console.log("OK " + checks + " checks");
