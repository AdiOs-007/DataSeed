"""Futuristic interactive background for the DataSeed Streamlit UI."""

_HTML = r"""
<!doctype html>
<html>
<body>
<script>
(() => {
  let win, doc;

  try {
    win = window.parent;
    doc = win.document;
  } catch (_) {
    return;
  }

  // Streamlit reruns the script after interactions; install only once.
  if (doc.getElementById("dataseed-neural-canvas")) return;

  const frame = window.frameElement;
  if (frame) {
    frame.style.cssText =
      "position:fixed!important;left:0!important;top:0!important;" +
      "width:1px!important;height:1px!important;opacity:0!important;" +
      "pointer-events:none!important;border:0!important;z-index:-1!important;";
  }

  const style = doc.createElement("style");
  style.id = "dataseed-neural-style";
  style.textContent = `
    html, body { background: transparent !important; }
    body { background-color: #070b10 !important; }

    #root {
      position: relative !important;
      z-index: 1 !important;
      background: transparent !important;
    }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    section.main {
      background: transparent !important;
    }

    [data-testid="stHeader"] {
      background: rgba(7, 11, 16, .70) !important;
      backdrop-filter: blur(12px);
    }

    [data-testid="stSidebar"] {
      background: rgba(7, 12, 19, .88) !important;
      border-right: 1px solid rgba(65, 255, 177, .18);
      backdrop-filter: blur(14px);
    }

    [data-testid="stSidebar"] > div {
      background: transparent !important;
    }

    .stButton > button,
    [data-testid="stDownloadButton"] button {
      border: 1px solid rgba(65, 255, 177, .32) !important;
      background: rgba(11, 24, 29, .78) !important;
      transition: border-color .2s, box-shadow .2s, transform .2s;
    }

    .stButton > button:hover,
    [data-testid="stDownloadButton"] button:hover {
      border-color: #43ffc0 !important;
      box-shadow: 0 0 16px rgba(48, 255, 183, .20);
      transform: translateY(-1px);
    }

    .stButton > button[kind="primary"] {
      background: linear-gradient(105deg, #075a49, #087e73) !important;
      border-color: #3dffc0 !important;
      box-shadow: 0 0 16px rgba(48, 255, 183, .12);
    }

    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stNumberInput"] input {
      background: rgba(7, 15, 23, .90) !important;
      border-color: rgba(74, 220, 194, .30) !important;
    }

    [data-testid="stTextInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus,
    [data-testid="stNumberInput"] input:focus {
      border-color: #43ffc0 !important;
      box-shadow: 0 0 0 1px rgba(67, 255, 192, .38) !important;
    }

    div[data-testid="stMetric"] {
      background: rgba(11, 22, 30, .72);
      border: 1px solid rgba(65, 255, 177, .18);
      border-radius: 12px;
      padding: 14px;
    }

    #dataseed-neural-canvas {
      position: fixed !important;
      inset: 0 !important;
      width: 100vw !important;
      height: 100vh !important;
      pointer-events: none !important;
      z-index: 0 !important;
    }
  `;
  doc.head.appendChild(style);

  const canvas = doc.createElement("canvas");
  canvas.id = "dataseed-neural-canvas";
  canvas.setAttribute("aria-hidden", "true");
  doc.body.appendChild(canvas);

  const ctx = canvas.getContext("2d", { alpha: true });
  if (!ctx) return;

  const symbols =
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" +
    "{}[]<>/$%+=*&!?;:~^_|πΣΔλ∞∑√";

  const reduced = win.matchMedia &&
    win.matchMedia("(prefers-reduced-motion: reduce)").matches;

  let width = 0, height = 0, dpr = 1;
  let particles = [];
  let pulses = [];
  let mouse = { x: -9999, y: -9999, active: false };
  const pulseDuration = 850;

  const rand = (min, max) => Math.random() * (max - min) + min;

  function makeParticle() {
    return {
      x: rand(0, width),
      y: rand(0, height),
      vx: rand(-0.22, 0.22),
      vy: rand(-0.22, 0.22),
      char: symbols[Math.floor(Math.random() * symbols.length)],
      size: rand(11, 19),
      alpha: rand(0.13, 0.38),
      color: Math.random() > 0.48 ? "#4affae" : "#42dfff",
      phase: rand(0, Math.PI * 2)
    };
  }

  function resize() {
    width = win.innerWidth;
    height = win.innerHeight;
    dpr = Math.min(win.devicePixelRatio || 1, 2);

    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const count = Math.min(
      reduced ? 85 : 220,
      Math.max(reduced ? 45 : 85, Math.round(width * height / 6500))
    );

    while (particles.length < count) particles.push(makeParticle());
    particles.length = count;

    for (const p of particles) {
      p.x = Math.min(p.x, width);
      p.y = Math.min(p.y, height);
    }
  }

  win.addEventListener("resize", resize, { passive: true });

  doc.addEventListener("pointermove", e => {
    mouse.x = e.clientX;
    mouse.y = e.clientY;
    mouse.active = true;
  }, { passive: true });

  doc.addEventListener("pointerleave", () => {
    mouse.active = false;
  }, { passive: true });

  // Capture clicks anywhere: buttons, text inputs, tables and the page.
  doc.addEventListener("click", e => {
    pulses.push({ x: e.clientX, y: e.clientY, start: Date.now() });
    if (pulses.length > 5) pulses.shift();
  }, true);

  function animate() {
    const now = Date.now();
    ctx.clearRect(0, 0, width, height);

    pulses = pulses.filter(p => now - p.start < pulseDuration);

    let globalPulse = 0;
    for (const pulse of pulses) {
      const progress = (now - pulse.start) / pulseDuration;
      globalPulse = Math.max(
        globalPulse,
        Math.sin(progress * Math.PI) * (1 - progress * 0.38)
      );
    }

    // Move characters and gently repel them from the cursor.
    for (const p of particles) {
      p.phase += 0.006;

      if (mouse.active && !reduced) {
        const dx = p.x - mouse.x;
        const dy = p.y - mouse.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const radius = 125;

        if (dist < radius) {
          const force = (1 - dist / radius) * 0.065;
          p.vx += (dx / dist) * force;
          p.vy += (dy / dist) * force;
        }
      }

      p.vx = Math.max(-0.8, Math.min(0.8, p.vx * 0.994));
      p.vy = Math.max(-0.8, Math.min(0.8, p.vy * 0.994));

      p.x += p.vx;
      p.y += p.vy;

      if (p.x < -15) p.x = width + 15;
      if (p.x > width + 15) p.x = -15;
      if (p.y < -15) p.y = height + 15;
      if (p.y > height + 15) p.y = -15;
    }

    // Fine connecting lines give the characters a data-network feel.
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const a = particles[i], b = particles[j];
        const dx = a.x - b.x, dy = a.y - b.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < 105) {
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.strokeStyle = `rgba(57, 239, 194, ${
            (1 - dist / 105) * (0.075 + globalPulse * 0.20)
          })`;
          ctx.lineWidth = 0.6 + globalPulse * 0.5;
          ctx.stroke();
        }
      }
    }

    // An expanding energy ring follows each click.
    for (const pulse of pulses) {
      const progress = (now - pulse.start) / pulseDuration;
      const radius = progress * Math.max(width, height) * 0.80;
      const fade = 1 - progress;

      ctx.beginPath();
      ctx.arc(pulse.x, pulse.y, radius, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(71, 255, 192, ${fade * 0.38})`;
      ctx.lineWidth = 1 + fade * 2;
      ctx.shadowColor = "#4affd0";
      ctx.shadowBlur = 8 * fade;
      ctx.stroke();
      ctx.shadowBlur = 0;
    }

    // Render glowing characters; every click briefly lights the whole field.
    for (const p of particles) {
      let localPulse = 0;

      for (const pulse of pulses) {
        const age = now - pulse.start;
        const progress = age / pulseDuration;
        const waveRadius = progress * Math.max(width, height) * 0.80;
        const dist = Math.hypot(p.x - pulse.x, p.y - pulse.y);
        const wave = Math.max(0, 1 - Math.abs(dist - waveRadius) / 100);
        const waveStrength = wave * (1 - progress);
        localPulse = Math.max(localPulse, waveStrength);
      }

      const glow = Math.min(1, globalPulse * 0.95 + localPulse * 1.5);
      const alpha = Math.min(1, p.alpha + glow * 0.66);
      const size = p.size + glow * 5;

      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.fillStyle = p.color;
      ctx.shadowColor = p.color;
      ctx.shadowBlur = 2 + glow * 19;
      ctx.font = `${glow > 0.35 ? "600 " : ""}${size}px monospace`;
      ctx.fillText(
        p.char,
        p.x + Math.sin(p.phase) * 2,
        p.y + Math.cos(p.phase) * 2
      );
      ctx.restore();
    }

    win.requestAnimationFrame(animate);
  }

  resize();
  win.requestAnimationFrame(animate);
})();
</script>
</body>
</html>
"""


def inject_neural_background() -> None:
    """Install the background in the parent Streamlit page without blocking UI."""
    try:
        import streamlit.components.v1 as components
        components.html(_HTML, height=1, scrolling=False)
    except Exception:
        # Decorative effects should never prevent the actual app from working.
        pass
