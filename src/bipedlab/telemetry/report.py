"""Self-contained HTML report from a telemetry recording (backlog WP2-02).

No plotting dependency: charts are inline SVG polylines, so the file opens
offline in any browser and is small enough to keep as a CI artifact.
Labels are Turkish (owner-facing); channel names stay as in the schema.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np

from bipedlab.telemetry import recorder as rec

PALETTE = ["#2f6fdf", "#e0702b", "#2a9d6f", "#c2417d", "#7a5bd6", "#8a8f3a", "#3aa0b8"]
W, H, PAD = 720, 200, 36


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "Evet" if value else "Hayır"
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, list):
        return ", ".join(_fmt(v) for v in value)
    return html.escape(str(value))


def _chart(title: str, note: str, t: np.ndarray, series: dict[str, np.ndarray], unit: str) -> str:
    finite = [s[np.isfinite(s)] for s in series.values()]
    lo = min(float(s.min()) for s in finite)
    hi = max(float(s.max()) for s in finite)
    if hi - lo < 1e-9:
        lo, hi = lo - 1.0, hi + 1.0
    t0, t1 = float(t[0]), float(t[-1]) if t[-1] > t[0] else float(t[0]) + 1.0

    def x(v: np.ndarray) -> np.ndarray:
        return PAD + (v - t0) / (t1 - t0) * (W - 2 * PAD)

    def y(v: np.ndarray) -> np.ndarray:
        return H - PAD + (v - lo) / (hi - lo) * -(H - 2 * PAD)

    lines, legend = [], []
    for i, (name, values) in enumerate(series.items()):
        color = PALETTE[i % len(PALETTE)]
        pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(x(t), y(values), strict=True))
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.6" points="{pts}"/>')
        legend.append(f'<span><i style="background:{color}"></i>{html.escape(name)}</span>')
    zero = ""
    if lo < 0 < hi:
        zy = float(y(np.array(0.0)))
        zero = f'<line x1="{PAD}" x2="{W - PAD}" y1="{zy:.1f}" y2="{zy:.1f}" class="zero"/>'
    axes = (
        f'<text x="{PAD - 4}" y="{PAD}" text-anchor="end">{hi:.3g}</text>'
        f'<text x="{PAD - 4}" y="{H - PAD}" text-anchor="end">{lo:.3g}</text>'
        f'<text x="{PAD}" y="{H - 8}">{t0:.1f} s</text>'
        f'<text x="{W - PAD}" y="{H - 8}" text-anchor="end">{t1:.1f} s</text>'
        f'<text x="{W - PAD}" y="{PAD - 10}" text-anchor="end">{html.escape(unit)}</text>'
    )
    return (
        f"<section><h2>{html.escape(title)}</h2><p>{html.escape(note)}</p>"
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">'
        f'<rect x="{PAD}" y="{PAD}" width="{W - 2 * PAD}" height="{H - 2 * PAD}" class="frame"/>'
        f"{zero}{''.join(lines)}{axes}</svg>"
        f'<div class="legend">{"".join(legend)}</div></section>'
    )


def _histogram(counts: np.ndarray, edges: np.ndarray, deadline_ms: float) -> str:
    """Bar chart of step wall time with the control deadline marked."""
    lo, hi = float(edges[0]), float(edges[-1])
    top = max(int(counts.max()), 1)
    span = max(hi - lo, 1e-6)

    def x(v: float) -> float:
        return PAD + (v - lo) / span * (W - 2 * PAD)

    bars = "".join(
        f'<rect x="{x(float(a)):.1f}" y="{H - PAD - (c / top) * (H - 2 * PAD):.1f}" '
        f'width="{max(x(float(b)) - x(float(a)) - 1, 1):.1f}" '
        f'height="{(c / top) * (H - 2 * PAD):.1f}" fill="{PALETTE[0]}"/>'
        for a, b, c in zip(edges[:-1], edges[1:], counts, strict=True)
    )
    marker = ""
    if lo <= deadline_ms <= hi:
        dx = x(deadline_ms)
        marker = (
            f'<line x1="{dx:.1f}" x2="{dx:.1f}" y1="{PAD}" y2="{H - PAD}" '
            f'stroke="{PALETTE[1]}" stroke-dasharray="4 3"/>'
            f'<text x="{dx + 4:.1f}" y="{PAD + 12}">{deadline_ms:.0f} ms sınırı</text>'
        )
    axes = (
        f'<text x="{PAD - 4}" y="{PAD}" text-anchor="end">{top}</text>'
        f'<text x="{PAD - 4}" y="{H - PAD}" text-anchor="end">0</text>'
        f'<text x="{PAD}" y="{H - 8}">{lo:.3g} ms</text>'
        f'<text x="{W - PAD}" y="{H - 8}" text-anchor="end">{hi:.3g} ms</text>'
    )
    title = "Adım süresi dağılımı"
    return (
        f"<section><h2>{title}</h2><p>Kaç adımın ne kadar sürdüğü. Hepsi kesikli çizginin "
        f"solundaysa bu makine 50 Hz'i gerçek zamanda yetiştirebilir.</p>"
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{title}">'
        f'<rect x="{PAD}" y="{PAD}" width="{W - 2 * PAD}" height="{H - 2 * PAD}" class="frame"/>'
        f"{bars}{marker}{axes}</svg></section>"
    )


SUMMARY_LABELS = {
    "fell": ("Düştü mü?", "Gövde 60°'den fazla eğildiyse evet."),
    "first_fall_s": ("İlk düşme anı", "saniye"),
    "max_tilt_deg": ("En büyük eğim", "derece; 0 = tam dik"),
    "mean_vx_body_mps": ("Ortalama ileri hız", "m/s, robotun kendi önüne göre"),
    "vx_tracking_error_mps": (
        "İleri hız hatası",
        "komut ile gerçekleşen arasındaki ortalama fark, m/s",
    ),
    "mean_yaw_rate_rps": ("Ortalama dönüş hızı", "rad/s; pozitif = sola"),
    "yaw_tracking_error_rps": ("Dönüş hızı hatası", "rad/s"),
    "mean_power_w": ("Ortalama mekanik güç", "W; motorların harcadığı efor"),
    "peak_abs_torque_nm": ("En yüksek tork", "N·m; motorun en zorlandığı an"),
    "foot_contact_ratio": ("Ayak yerde oranı (sol, sağ)", "1 = hep yerde; yürürken < 1 olmalı"),
    "realtime_factor": ("Gerçek zamana göre hız", "kaç kat hızlı simüle edildi (bu makinede)"),
    "wall_step_ms_p50": ("Adım süresi p50", "ms; tipik bir kontrol adımının hesap süresi"),
    "wall_step_ms_p99": ("Adım süresi p99", "ms; kontrol döngüsünün en yavaş %1'i"),
    "wall_step_jitter_ms": ("Titreşim (p99 − p50)", "ms; küçükse döngü düzenli"),
    "deadline_miss_ratio": (
        "Süre aşımı oranı",
        "20 ms'den uzun süren adımların payı; gerçek zamanlı 50 Hz için 0 olmalı",
    ),
    "sim_period_max_error_ms": (
        "Simülasyon periyot hatası",
        "ms; adımlar arası simülasyon zamanı 20 ms'den ne kadar saptı (0 olmalı)",
    ),
    "steps": ("Kontrol adımı sayısı", "50 Hz"),
}


def render(arrays: dict[str, np.ndarray], sidecar: dict[str, Any], title: str) -> str:
    t = arrays["time"]
    summary = rec.summarize(arrays)
    meta = sidecar.get("meta", {})
    rows = "".join(
        f"<tr><th>{label}</th><td>{_fmt(summary[key])}</td><td class='note'>{note}</td></tr>"
        for key, (label, note) in SUMMARY_LABELS.items()
        if key in summary
    )
    meta_rows = "".join(
        f"<tr><th>{html.escape(str(k))}</th><td colspan=2>{_fmt(v)}</td></tr>"
        for k, v in meta.items()
    )
    cmd = arrays["command"]
    charts = [
        _chart(
            "Gövde eğimi",
            "Dik durmaktan sapma. 60° üstü düşme sayılır.",
            t,
            {"eğim": arrays["tilt"]},
            "derece",
        ),
    ]
    if cmd.shape[-1] >= 3:
        charts.append(
            _chart(
                "İleri hız: komut ve gerçekleşen",
                "İki çizgi üst üste ise robot komutu izliyor.",
                t,
                {"komut vx": cmd[:, 0], "ölçülen vx": arrays["lin_vel_body"][:, 0]},
                "m/s",
            )
        )
        charts.append(
            _chart(
                "Dönüş hızı: komut ve gerçekleşen",
                "Pozitif = sola dönüş.",
                t,
                {"komut wz": cmd[:, 2], "ölçülen wz": arrays["gyro"][:, 2]},
                "rad/s",
            )
        )
    charts.append(
        _chart(
            "Mekanik güç", "Σ|tork × açısal hız|. Yürürken artar.", t, {"güç": arrays["power"]}, "W"
        )
    )
    contact = arrays["foot_contact"].astype(float)
    charts.append(
        _chart(
            "Ayak teması",
            "1 = ayak yerde, 0 = havada. Yürürken sırayla değişmeli.",
            t,
            {"sol": contact[:, 0] + 0.02, "sağ": contact[:, 1] - 0.02},
            "0/1",
        )
    )
    n_legs = min(5, arrays["joint_pos"].shape[1])
    charts.append(
        _chart(
            "Sol bacak eklemleri",
            "İlk 5 eklemin açısı.",
            t,
            {f"eklem {i}": arrays["joint_pos"][:, i] for i in range(n_legs)},
            "rad",
        )
    )
    charts.append(
        _chart(
            "En yüksek eklem torku",
            "Her adımda en çok zorlanan motorun torku.",
            t,
            {"|tork| maks": np.abs(arrays["joint_torque"]).max(axis=1)},
            "N·m",
        )
    )
    if len(t) > 1:
        charts.append(
            _chart(
                "Kontrol adımı süresi",
                "Bilgisayarın bir adımı hesaplama süresi.",
                t[1:],
                {"adım süresi": arrays["wall_dt"][1:] * 1e3},
                "ms",
            )
        )
    counts, edges = rec.wall_histogram(arrays)
    if len(counts):
        charts.append(_histogram(counts, edges, 1e3 / 50.0))
    return f"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root{{--bg:#fbfbfa;--fg:#1d1f23;--muted:#666b73;--line:#d9dbe0;--card:#fff}}
@media (prefers-color-scheme:dark){{:root{{--bg:#16181c;--fg:#e8e9eb;--muted:#9aa0a8;--line:#30343b;--card:#1e2126}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:16px}}
main{{max-width:760px;margin:auto}} h1{{font-size:22px}} h2{{font-size:16px;margin:0 0 2px}}
p{{color:var(--muted);margin:0 0 6px}} section,table{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin:12px 0}}
table{{border-collapse:collapse;width:100%}} th,td{{text-align:left;padding:4px 8px;border-bottom:1px solid var(--line);vertical-align:top}}
td.note{{color:var(--muted);font-size:13px}} svg{{width:100%;height:auto}} svg text{{fill:var(--muted);font-size:11px}}
.frame{{fill:none;stroke:var(--line)}} .zero{{stroke:var(--line);stroke-dasharray:4 3}}
.legend span{{margin-right:14px;font-size:13px}} .legend i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px}}
</style></head><body><main>
<h1>{html.escape(title)}</h1>
<p>Simülasyon kaydı. Ortam etiketi bu sonucun nerede üretildiğini söyler; başka ortam için kanıt sayılmaz.</p>
<table>{meta_rows}</table>
<table>{rows}</table>
{"".join(charts)}
</main></body></html>
"""


def write(npz_path: str | Path, out: str | Path | None = None, title: str | None = None) -> Path:
    arrays, sidecar = rec.load(npz_path)
    out = Path(out) if out else Path(npz_path).with_suffix(".html")
    name = sidecar.get("meta", {}).get("policy", Path(npz_path).stem)
    out.write_text(render(arrays, sidecar, title or f"Telemetri raporu: {name}"))
    return out


def main_json(npz_path: str | Path) -> str:
    arrays, _ = rec.load(npz_path)
    return json.dumps(rec.summarize(arrays), indent=2)
