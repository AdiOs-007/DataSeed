"""Safety tests for the decorative visual layer (theme CSS, loader, fallbacks)."""
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

from dataseed import visual_theme as vt

ROOT = Path(__file__).resolve().parent.parent
JS = ROOT / "tests" / "js"


def css_rules(css):
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    return [(m.group(1).strip(), m.group(2)) for m in re.finditer(r"([^{}@]+)\{([^{}]*)\}", css)]


def test_theme_css_never_hides_moves_or_restacks_the_app():
    css = vt.theme_css()
    flat = re.sub(r"\s+", "", css)
    for forbidden in ("visibility:hidden", "position:fixed", "z-index", "pointer-events", "!importantdisplay"):
        assert forbidden not in flat, forbidden
    # nothing but the helper containers may use display:none
    for sel, body in css_rules(css):
        if "display:none" in body.replace(" ", ""):
            assert "#ds-style-anchor" in sel, sel
        assert not re.search(r"(^|,)\s*(\.stApp|body|html)\s*(,|$)", sel) or "display" not in body, sel


def test_transparent_backgrounds_apply_only_while_the_canvas_is_mounted():
    transparent = [(s, b) for s, b in css_rules(vt.theme_css()) if re.search(r"background:\s*transparent", b)]
    assert transparent
    for sel, _ in transparent:
        assert all(part.strip().startswith('html[data-ds-fx="on"]') for part in sel.split(",")), sel


def test_engine_file_is_present_packaged_and_safe_to_embed():
    js = vt.engine_js()
    assert js and "__dataseedFx" in js and vt.engine_version(js) == "1.0.0"
    assert "fx/*.js" in (ROOT / "pyproject.toml").read_text()
    assert "preventDefault" not in js.replace("never preventDefault", "") or "preventDefault()" not in js
    assert "stopPropagation()" not in js
    html = vt.loader_html("Full")
    assert html.count("</script>") == 1 and "window.parent" in html and "catch" in html
    assert '"mode": "full"' in vt.loader_html("Full") and '"mode": "off"' in vt.loader_html("Off")
    assert '"mode": "full"' in vt.loader_html("nonsense")


def test_default_mode_from_environment(monkeypatch=None):
    import os
    old = os.environ.get("DATASEED_FX")
    try:
        for val, want in (("off", "Off"), ("CALM", "Calm"), ("full", "Full"), ("junk", "Full"), ("", "Full")):
            os.environ["DATASEED_FX"] = val
            assert vt.default_mode() == want
        del os.environ["DATASEED_FX"]
        assert vt.default_mode() == "Full"
    finally:
        if old is not None:
            os.environ["DATASEED_FX"] = old
        else:
            os.environ.pop("DATASEED_FX", None)


def test_emitted_markdown_has_no_newlines_so_markdown_cannot_split_the_style_block():
    md = []
    with_modules(fake_streamlit(md_calls=md), lambda: vt.apply("Full"))
    body = md[0][0]
    assert "\n" not in body and "/*" not in body and body.startswith("<span") and body.endswith("</style>")
    assert "\n" not in vt.hero_html("1.0")
    assert vt.theme_css().count("{") == vt.theme_css().count("}") > 40


def test_hero_markup():
    h = vt.hero_html("9.9")
    assert h.count("<span style=") == 14 and "v9.9" in h and 'aria-hidden="true"' in h


def fake_streamlit(markdown_ok=True, html_ok=True, html_calls=None, md_calls=None):
    st = types.ModuleType("streamlit")
    def markdown(body, **kw):
        if not markdown_ok:
            raise RuntimeError("markdown broke")
        if md_calls is not None:
            md_calls.append((body, kw))
    st.markdown = markdown
    comp = types.ModuleType("streamlit.components.v1")
    def html(body, height=None, **kw):
        if html_calls is not None:
            html_calls.append((body, height))
        if not html_ok:
            raise RuntimeError("iframe broke")
    comp.html = html
    pkg = types.ModuleType("streamlit.components"); pkg.v1 = comp
    st.components = pkg
    return {"streamlit": st, "streamlit.components": pkg, "streamlit.components.v1": comp}


def with_modules(mods, fn):
    saved = {k: sys.modules.get(k) for k in mods}
    sys.modules.update(mods)
    try:
        return fn()
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_apply_emits_theme_and_loader_and_never_raises():
    md, calls = [], []
    assert with_modules(fake_streamlit(md_calls=md, html_calls=calls), lambda: vt.apply("Calm")) is True
    assert md and md[0][1].get("unsafe_allow_html") is True and "<style>" in md[0][0]
    assert len(calls) == 1 and calls[0][1] == 0 and '"mode": "calm"' in calls[0][0]
    assert with_modules(fake_streamlit(html_ok=False), lambda: vt.apply("Full")) is True      # iframe failure: theme still applied
    assert with_modules(fake_streamlit(markdown_ok=False), lambda: vt.apply("Full")) is False  # CSS failure: reported, no exception
    assert with_modules({"streamlit": None}, lambda: vt.apply("Full")) is False               # streamlit missing: no exception


def test_missing_engine_file_means_theme_only():
    calls = []
    old = vt.ENGINE_PATH
    vt.ENGINE_PATH = ROOT / "does-not-exist.js"
    try:
        assert with_modules(fake_streamlit(html_calls=calls), lambda: vt.apply("Full")) is True
    finally:
        vt.ENGINE_PATH = old
    assert calls == []


def test_app_still_runs_when_the_visual_module_cannot_be_imported():
    from fake_streamlit import run
    saved = sys.modules.get("dataseed.visual_theme")
    sys.modules["dataseed.visual_theme"] = None          # makes `from dataseed import visual_theme` raise ImportError
    try:
        st, _ = run({"Load: Student schema"}, 1)
        assert '"student_id"' in st.session_state.schema_text
    finally:
        if saved is not None:
            sys.modules["dataseed.visual_theme"] = saved
        else:
            sys.modules.pop("dataseed.visual_theme", None)


def test_engine_logic_in_node(tmp_path=None):
    if shutil.which("node") is None:
        return
    r = subprocess.run(["node", str(JS / "fx_engine_harness.js")], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and r.stdout.startswith("OK"), r.stdout + r.stderr


def test_loader_logic_in_node(tmp_path):
    if shutil.which("node") is None:
        return
    full, off = tmp_path / "full.html", tmp_path / "off.html"
    full.write_text(vt.loader_html("Full"), encoding="utf-8")
    off.write_text(vt.loader_html("Off"), encoding="utf-8")
    r = subprocess.run(["node", str(JS / "loader_harness.js"), str(full), str(off)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and r.stdout.startswith("OK"), r.stdout + r.stderr
