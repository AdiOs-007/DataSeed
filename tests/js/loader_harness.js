/* Executes the loader HTML's <script> (as produced by dataseed.visual_theme.loader_html) inside a fake iframe whose
 * `parent` is a fake page. Usage: node loader_harness.js <loader-html-file-for-full> <loader-html-for-off>. */
"use strict";
const fs = require("fs");
const vm = require("vm");
const { makeEnv } = require("./fake_page");
let checks = 0;
function ok(c, m) { checks++; if (!c) { console.error("FAIL: " + m); process.exit(1); } }
function scriptOf(file) {
  const html = fs.readFileSync(file, "utf8");
  ok((html.match(/<\/script>/g) || []).length === 1, "loader contains exactly one closing script tag (engine text is escaped)");
  return html.slice(html.indexOf("<script>") + 8, html.lastIndexOf("</script>"));
}
function runLoader(code, iframeWin) { vm.runInContext(code, vm.createContext({ window: iframeWin, JSON })); }

const full = scriptOf(process.argv[2]), off = scriptOf(process.argv[3]);

// 1. normal page: engine injected once, configured, idempotent across reruns
let P = makeEnv(); const iframe = { parent: P.win };
runLoader(full, iframe);
ok(P.fx() && P.fx().stats().mounted && P.fx().stats().running, "loader injects and starts the engine");
ok(P.body.children.length === 1 && P.scriptEvals === 1, "one canvas, engine evaluated once");
const listeners = P.regs.length;
for (let i = 0; i < 5; i++) runLoader(full, { parent: P.win });
ok(P.body.children.length === 1 && P.scriptEvals === 1 && P.regs.length === listeners, "5 more reruns of the loader (new iframes) duplicate nothing");
P.advance(160); ok(P.fx().stats().frames > 5, "animation runs");
// 2. switching to Off from a later iframe removes everything; back to full restores
runLoader(off, { parent: P.win });
ok(P.body.children.length === 0 && P.regs.length === 0 && !P.attrs["data-ds-fx"], "Off removes canvas, listeners and marker");
runLoader(full, { parent: P.win });
ok(P.body.children.length === 1 && P.regs.length === listeners && P.scriptEvals === 1, "back to Full without re-injecting the engine");
// 3. parent not accessible (cross-origin sandbox): loader must not throw and must do nothing
const blocked = { get parent() { throw new Error("SecurityError: blocked a frame from accessing a cross-origin frame"); } };
let threw = false; try { runLoader(full, blocked); } catch (e) { threw = true; }
ok(!threw, "inaccessible parent: loader fails silently");
const blockedDoc = { parent: { get document() { throw new Error("SecurityError"); } } };
threw = false; try { runLoader(full, blockedDoc); } catch (e) { threw = true; }
ok(!threw, "inaccessible parent document: loader fails silently");
// 4. an older engine version is replaced
const Q = makeEnv(); let destroyed = false;
Q.win.__dataseedFx = { version: "0.0.1", destroy() { destroyed = true; Q.win.__dataseedFx = null; } };
runLoader(full, { parent: Q.win });
ok(destroyed && Q.fx().version !== "0.0.1" && Q.fx().stats().mounted, "older engine destroyed and replaced");
console.log("OK " + checks + " checks");
