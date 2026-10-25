# DataSeed visual theme & animated background

Purely decorative layer. Files: `src/dataseed/visual_theme.py` (CSS + loader), `src/dataseed/fx/engine.js` (canvas engine),
`.streamlit/config.toml` (colour theme), tests in `tests/test_visual_theme.py` and `tests/js/`.

## How it is integrated (and why)
| Piece | Mechanism | Risk |
|---|---|---|
| Colours / fonts / panels / buttons / tabs / inputs | `st.markdown("<style>...", unsafe_allow_html=True)` + Streamlit `[theme]` options (`config.toml` or `--theme.*` flags set by `python -m dataseed ui`) | CSS selectors target `data-testid` attributes; a selector that stops matching in a future Streamlit simply loses that styling |
| Animated background | One `<canvas id="dataseed-fx-canvas">` with `position:fixed; z-index:-1; pointer-events:none`, drawn by one `requestAnimationFrame` loop | Layering uses a negative z-index, so it is *behind* all content and never in front |
| Cursor + click events | Passive, capture-phase listeners on `document` (no `preventDefault`/`stopPropagation`) | Needs page-level events, see below |
| Getting the engine into the page | A zero-height `components.html` iframe runs a loader that injects `engine.js` once into the parent page | **Relies on the component iframe being same-origin with the page** (true for standard Streamlit; not verified for every version) |

A Streamlit component iframe cannot see mouse movement or clicks outside itself. The only way to react to the cursor and to clicks
anywhere on the page is for code to run in the page itself; that is why the loader injects the engine (once, namespaced, removable).
If the iframe cannot reach the page, the loader does nothing and the app keeps its normal look (theme colours still apply).

## Safety properties (all covered by tests)
* Singleton `window.__dataseedFx`: Streamlit reruns, extra iframes or repeated loads never add canvases, loops or listeners.
* `<html data-ds-fx="on">` exists only while the canvas exists; the CSS makes Streamlit backgrounds transparent only under that attribute.
* The engine refuses to mount if `<body>` is a stacking context (the canvas would be invisible) and removes itself after 3 consecutive draw errors.
* Click pulse: global glow + ripple, ends at 850 ms (calm mode 600 ms); pulses capped at 6; no timers per click.
* Cursor: nearby particles are pushed away with a slight swirl by a damped spring and settle back after the cursor leaves.
* `prefers-reduced-motion` or **Background effects = Calm**: ~55 % particles, slow drift, no cursor effect, soft glow only.
* Pauses when the tab is hidden; canvas resolution capped at 1.5x DPR; at most 120 particles; link lines only among ~1/3 of particles and capped at 140; adaptive shedding of particles if frames get slow.

## Switching it off
* Sidebar: **Background effects → Off**.
* Environment: `DATASEED_FX=off python -m dataseed ui` (or `calm`).
* Remove the two `visual_theme` blocks from `streamlit_app.py`; nothing else depends on this layer.

## Debugging in the browser (DevTools console)
```js
document.getElementById("dataseed-fx-canvas")   // should exist (one)
window.__dataseedFx.stats()                      // {mode, mounted, running, particles, pulses, listeners:7, frames, costMs}
document.documentElement.getAttribute("data-ds-fx")  // "on"
```
If the canvas exists but nothing is visible, some Streamlit container still has an opaque background: inspect the element stack at a blank
spot and add its selector to the "transparent" rule list in `THEME_CSS`. If `window.__dataseedFx` is undefined, the iframe could not reach the page.

## Manual browser checklist
Sidebar and main area both show the field; Generate dataset still works while the pulse plays; clicking buttons, text areas, radios,
file-upload and the table all pulse; the pulse is gone in under 1 s; rapid clicking does not stutter; scroll and Tab-key navigation work;
after switching input methods the particle count in `stats()` does not grow; Background effects = Off removes the canvas.
