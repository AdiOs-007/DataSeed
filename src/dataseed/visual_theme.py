"""Decorative visual layer for the Streamlit UI: futuristic theme + animated data-particle background.

SAFETY MODEL (read before changing anything)
    * The theme is plain CSS emitted with ``st.markdown`` (a supported mechanism). It never hides, moves
      or re-stacks the Streamlit app; the only global rules set colours/borders/typography.
    * The animated background is a single ``<canvas>`` at ``z-index:-1`` created by ``fx/engine.js``.
      Streamlit app containers are made transparent ONLY while ``<html data-ds-fx="on">`` is set, i.e.
      only while the canvas really exists. If the engine cannot start, the app looks normal.
    * Cursor/click interaction needs page-level events, which a Streamlit component iframe cannot see by
      itself. The loader below therefore (a) checks that the iframe may access the parent page, (b)
      injects the engine script once into the parent page, which then owns the canvas and passive
      listeners (it never calls preventDefault/stopPropagation). If the parent is not accessible the
      loader silently does nothing.
    * Every public function swallows its own errors: a failure here must never take the app down.
    * Kill switches: sidebar "Background effects" = Off, or environment variable ``DATASEED_FX=off``.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

log = logging.getLogger("dataseed.visual_theme")

FX_MODES = ("Full", "Calm", "Off")
ENGINE_PATH = Path(__file__).with_name("fx") / "engine.js"

THEME_CSS = r"""
:root{--ds-bg:#05080f;--ds-panel:rgba(9,16,28,.74);--ds-panel-solid:#0a1220;--ds-line:rgba(110,255,200,.16);
--ds-line-strong:rgba(110,255,200,.38);--ds-mint:#6effc8;--ds-cyan:#46ebff;--ds-text:#d8e6ee;--ds-muted:#8aa0ad;
--ds-mono:ui-monospace,"JetBrains Mono","Fira Code",Menlo,Consolas,monospace;}
html,body{background:var(--ds-bg)!important;}
.stApp{color:var(--ds-text);font-family:Inter,"Segoe UI",system-ui,-apple-system,sans-serif;}

/* background layers become transparent only while the animated canvas is mounted */
html[data-ds-fx="on"] .stApp,
html[data-ds-fx="on"] [data-testid="stApp"],
html[data-ds-fx="on"] [data-testid="stAppViewContainer"],
html[data-ds-fx="on"] [data-testid="stMain"],
html[data-ds-fx="on"] [data-testid="stHeader"]{background:transparent!important;}

/* sidebar: glass panel */
[data-testid="stSidebar"]{border-right:1px solid var(--ds-line);}
html[data-ds-fx="on"] [data-testid="stSidebar"]{background:rgba(6,11,20,.80)!important;}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{color:var(--ds-mint);}

/* title + headings */
.stApp h1{background:linear-gradient(90deg,var(--ds-mint),var(--ds-cyan));-webkit-background-clip:text;background-clip:text;
-webkit-text-fill-color:transparent;font-weight:700;letter-spacing:-.01em;}
.stApp h2{letter-spacing:.01em;padding-bottom:.25rem;border-bottom:1px solid var(--ds-line);}
.stApp h3{color:var(--ds-mint);}
[data-testid="stCaptionContainer"]{color:var(--ds-muted);}

/* hero data-bars next to the title */
.ds-hero{display:flex;align-items:center;gap:.9rem;margin:-.25rem 0 .6rem;font-family:var(--ds-mono);font-size:.72rem;color:var(--ds-muted);}
.ds-bars{display:flex;align-items:flex-end;gap:3px;height:22px;}
.ds-bars span{display:block;width:3px;height:22px;border-radius:2px;background:linear-gradient(to top,var(--ds-cyan),var(--ds-mint));
transform-origin:bottom;transform:scaleY(.25);opacity:.85;animation:ds-bar 1.9s ease-in-out infinite;animation-delay:calc(var(--i)*110ms);}
@keyframes ds-bar{0%,100%{transform:scaleY(.2)}45%{transform:scaleY(1)}70%{transform:scaleY(.45)}}

/* buttons */
.stApp button[kind="secondary"],.stApp [data-testid="stBaseButton-secondary"],.stApp [data-testid="stDownloadButton"] button{
background:rgba(10,20,34,.72);border:1px solid var(--ds-line);color:var(--ds-text);border-radius:8px;
transition:border-color .18s ease,box-shadow .18s ease,color .18s ease,background .18s ease;}
.stApp button[kind="secondary"]:hover,.stApp [data-testid="stBaseButton-secondary"]:hover,.stApp [data-testid="stDownloadButton"] button:hover{
border-color:var(--ds-line-strong);color:var(--ds-mint);box-shadow:0 0 0 1px rgba(110,255,200,.18),0 0 18px rgba(110,255,200,.10);}
.stApp button[kind="primary"],.stApp [data-testid="stBaseButton-primary"]{background:linear-gradient(90deg,#4df2b5,#35d8f2);color:#03110d;
border:0;border-radius:8px;font-weight:650;letter-spacing:.02em;box-shadow:0 0 22px rgba(70,235,255,.20);transition:box-shadow .18s ease,filter .18s ease;}
.stApp button[kind="primary"]:hover,.stApp [data-testid="stBaseButton-primary"]:hover{filter:brightness(1.07);box-shadow:0 0 30px rgba(110,255,200,.32);}
.stApp button:disabled{opacity:.42;}
.stApp button:focus-visible,.stApp [role="tab"]:focus-visible{outline:2px solid rgba(70,235,255,.75);outline-offset:2px;}

/* inputs */
.stApp [data-baseweb="input"]>div,.stApp [data-baseweb="select"]>div,.stApp [data-baseweb="textarea"],.stApp textarea{
background:rgba(8,14,26,.78)!important;border-color:var(--ds-line)!important;border-radius:8px;}
.stApp [data-baseweb="input"]:focus-within>div,.stApp [data-baseweb="select"]:focus-within>div,.stApp [data-baseweb="textarea"]:focus-within{
border-color:rgba(70,235,255,.65)!important;box-shadow:0 0 0 1px rgba(70,235,255,.35);}
.stApp textarea{font-family:var(--ds-mono);font-size:.82rem;}

/* panels */
[data-testid="stExpander"]{background:var(--ds-panel);border:1px solid var(--ds-line);border-radius:10px;}
[data-testid="stFileUploaderDropzone"]{background:var(--ds-panel);border:1px dashed rgba(110,255,200,.30);border-radius:10px;transition:border-color .18s ease,background .18s ease;}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--ds-mint);background:rgba(12,24,40,.82);}
[data-testid="stAlert"]{background:var(--ds-panel);border-radius:10px;}
[data-testid="stMetric"]{background:var(--ds-panel);border:1px solid var(--ds-line);border-radius:10px;padding:.7rem .9rem;}
[data-testid="stMetricValue"]{font-family:var(--ds-mono);color:var(--ds-mint);}
[data-testid="stMetricLabel"]{color:var(--ds-muted);}
[data-testid="stDataFrame"]{border:1px solid var(--ds-line);border-radius:10px;overflow:hidden;}

/* tabs */
[data-baseweb="tab-list"]{border-bottom:1px solid var(--ds-line);gap:.25rem;}
.stApp button[role="tab"][aria-selected="true"]{color:var(--ds-mint);}
[data-baseweb="tab-highlight"]{background:linear-gradient(90deg,var(--ds-mint),var(--ds-cyan))!important;}

/* thin scrollbars */
*{scrollbar-width:thin;scrollbar-color:rgba(110,255,200,.28) transparent;}

/* collapse the helper elements this module adds (the <style> holder and the zero-height loader iframe) */
[data-testid="stElementContainer"]:has(#ds-style-anchor),.element-container:has(#ds-style-anchor){display:none;}
[data-testid="stElementContainer"]:has(iframe[height="0"]),.element-container:has(iframe[height="0"]),
[data-testid="stElementContainer"]:has(iframe[height="1"]),.element-container:has(iframe[height="1"]){position:absolute;width:0;height:0;overflow:hidden;margin:0;padding:0;}

@media (prefers-reduced-motion:reduce){.ds-bars span{animation:none;transform:scaleY(.6);}
.stApp *{transition-duration:.01ms!important;}}
"""


def default_mode() -> str:
    """Initial effects mode: from ``DATASEED_FX`` (off | calm | full) or ``Full``."""
    v = os.environ.get("DATASEED_FX", "").strip().lower()
    return {"off": "Off", "calm": "Calm", "full": "Full"}.get(v, "Full")


def theme_css() -> str:
    """The theme as ONE line without comments.

    Streamlit renders ``st.markdown`` through a Markdown parser, and a blank line ends an HTML block: any CSS after the
    first blank line would be shown on the page as text. Keeping the CSS free of newlines avoids that.
    """
    css = re.sub(r"/\*.*?\*/", "", THEME_CSS, flags=re.S)
    return re.sub(r"\s*\n\s*", " ", css).strip()


def hero_html(version: str = "") -> str:
    bars = "".join(f'<span style="--i:{i}"></span>' for i in range(14))
    label = f"DATA LAB &middot; v{version}" if version else "DATA LAB"
    return f'<div class="ds-hero"><div class="ds-bars" aria-hidden="true">{bars}</div><span>{label}</span></div>'


def engine_js() -> str | None:
    try:
        return ENGINE_PATH.read_text(encoding="utf-8")
    except OSError:
        return None


def engine_version(js: str) -> str:
    m = re.search(r'VERSION\s*=\s*"([^"]+)"', js)
    return m.group(1) if m else "0"


def loader_html(mode: str) -> str | None:
    """HTML of the zero-height helper iframe. Returns None if the engine file is unavailable."""
    js = engine_js()
    if js is None:
        return None
    cfg = json.dumps({"mode": mode.lower() if mode.lower() in ("full", "calm", "off") else "full"})
    engine = json.dumps(js).replace("</", "<\\/")
    return f"""<!doctype html><html><head><meta charset="utf-8"></head><body><script>
(function () {{
  var ENGINE = {engine}, VERSION = {json.dumps(engine_version(js))}, CFG = {cfg};
  try {{
    var p = window.parent, d = p.document;            /* throws if the page is not accessible: then we do nothing */
    if (p.__dataseedFx && p.__dataseedFx.version !== VERSION) {{ p.__dataseedFx.destroy(); }}
    if (!p.__dataseedFx) {{
      var s = d.createElement("script");
      s.setAttribute("data-dataseed-fx", "1");
      s.textContent = ENGINE;
      d.head.appendChild(s);
      if (s.parentNode) {{ s.parentNode.removeChild(s); }}
    }}
    if (p.__dataseedFx) {{ p.__dataseedFx.configure(CFG); }}
  }} catch (e) {{ /* decorative only */ }}
}})();
</script></body></html>"""


def apply(mode: str | None = None) -> bool:
    """Emit theme CSS and (unless Off) start/configure the animated background. Never raises.

    Returns True if the theme CSS was emitted.
    """
    mode = mode if mode in FX_MODES else default_mode()
    try:
        import streamlit as st
        st.markdown(f'<span id="ds-style-anchor"></span><style>{theme_css()}</style>', unsafe_allow_html=True)
    except Exception as exc:  # pragma: no cover - depends on Streamlit
        log.warning("visual theme not applied: %s", exc)
        return False
    try:
        html = loader_html(mode)
        if html is not None:
            import streamlit.components.v1 as components
            try:
                components.html(html, height=0)
            except Exception:
                try:
                    components.html(html, height=1)
                except Exception:
                    # newer Streamlit versions deprecate components.html in favour of st.iframe (unverified here)
                    import streamlit as st
                    st.iframe(html, height=1)
    except Exception as exc:
        log.warning("animated background not started: %s", exc)
    return True


def hero(version: str = "") -> None:
    """Small animated data-bars strip under the title. Never raises."""
    try:
        import streamlit as st
        st.markdown(hero_html(version), unsafe_allow_html=True)
    except Exception as exc:  # pragma: no cover
        log.warning("hero not rendered: %s", exc)
