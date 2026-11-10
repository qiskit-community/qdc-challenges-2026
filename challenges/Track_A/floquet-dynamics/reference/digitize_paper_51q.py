"""Digitize the 51-qubit experimental data from the paper's Fig. 3.

The 51-qubit results are *error-mitigated hardware experiments* (arXiv:2607.24937v1,
Fig. 3) -- they are beyond exact statevector, so we cannot recompute them. To give
participants a concrete 51q target to aim their own hardware runs at, we digitize
them straight from the published figure. The values are therefore **approximate**
(pixel-level, +/- a few x1e-3) and are labelled as such wherever they are plotted.

What is extracted (51q only):
* panel (a): QESEM-Extrapolated ``M`` vs. Floquet cycle (the mitigated 51q dynamics);
* panel (b): coordination-two ``M_{z=2}`` vs. cycle over the late-cycle window 15-30;
* panel (c): the fitted oscillation amplitude at ``N_q = 51``.

Method: render page 6 at 300 dpi, calibrate each panel's axes from its tick marks
(constants below, read off the render), then for markers that sit at integer cycles
find the marker disk by its colour in a narrow vertical strip at each cycle (the
disk's widest scan-line is its centre; the thin fit line and error bars are
rejected). Colours are the marker RGBs sampled from the figure's own legend swatches.

    python -m reference.digitize_paper_51q         # -> reference/paper_fig3_51q.csv
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
PDF = HERE.parent / "doc" / "2607.24937v1.pdf"

# --- axis calibrations, read from the 300-dpi render of page 6 --------------- #
# panel (a): x(cycle) px and M(y) px
A_X = lambda c: 333.4 + 36.92 * c
A_M = lambda y: 0.54 + (y - 370.0) / (-2712.5)
A_Y = lambda M: 370.0 + (0.54 - M) * 2712.5
A_PINK = np.array([193, 61, 147])     # QESEM-Extrapolated marker (legend swatch)
A_YTOP, A_YBOT = 372, 1453

# panel (b): x(cycle) px and Mz2(y) px
B_X = lambda c: 1710.0 + 42.0 * (c - 15)
B_MZ2 = lambda y: 0.39 + (y - 425.5) / (-5464.3)
B_YMZ2 = lambda v: 425.5 + (v - 0.39) * (-5464.3)
B_BLUE = np.array([150, 190, 217])    # "51q exp." marker (legend swatch)
B_YTOP, B_YBOT = 372, 807

# panel (c): amplitude(y) px, purple experiment marker
C_AMP = lambda y: 0.045 - (y - 1030.0) / 9622.0
C_X51 = 1951                          # N_q = 51 tick (px)
C_PURPLE = np.array([95, 15, 117])
C_YTOP, C_YBOT = 1035, 1460


def render_page(dpi: int = 300) -> np.ndarray:
    """Render page 6 of the paper to an RGB array via pdftoppm."""
    from PIL import Image

    out = Path("/tmp/fig3page")
    subprocess.run(
        ["pdftoppm", "-png", "-r", str(dpi), "-f", "6", "-l", "6", str(PDF), str(out)],
        check=True, stderr=subprocess.DEVNULL,
    )
    return np.array(Image.open(f"{out}-06.png").convert("RGB")).astype(int)


def _disk_center_y(mask_strip: np.ndarray, y_off: int, min_peak: int = 6):
    """Centre-row (in full-image px) of the marker disk in a vertical colour strip.

    The disk's widest scan-line is its vertical centre; take the midpoint of the
    contiguous run of near-maximal rows around the peak. Returns ``None`` when the
    strip has too few coloured pixels (marker fully occluded)."""
    rc = mask_strip.sum(axis=1)
    if rc.max() < min_peak:
        return None
    a = int(rc.argmax())
    thr = 0.7 * rc.max()
    lo = hi = a
    while lo - 1 >= 0 and rc[lo - 1] >= thr:
        lo -= 1
    while hi + 1 < len(rc) and rc[hi + 1] >= thr:
        hi += 1
    return (lo + hi) / 2.0 + y_off


def digitize(im: np.ndarray) -> dict:
    """Return the digitized 51q data: panel-a M(cycle), panel-b Mz2(cycle), panel-c A."""
    def strip(target, x, ytop, ybot, half, tol):
        col = im[ytop:ybot, x - half:x + half + 1]
        return np.sqrt(((col - target) ** 2).sum(axis=2)) < tol

    # (a) QESEM-Extrapolated M at every integer cycle 1..30
    a_M = {}
    for c in range(1, 31):
        x = int(round(A_X(c)))
        yc = _disk_center_y(strip(A_PINK, x, A_YTOP, A_YBOT, 9, 60), A_YTOP)
        if yc is None:                                   # widen once if occluded
            yc = _disk_center_y(strip(A_PINK, x, A_YTOP, A_YBOT, 13, 60), A_YTOP)
        a_M[c] = None if yc is None else round(float(A_M(yc)), 4)

    # (b) 51q-exp Mz2 at every integer cycle 15..30
    b_Mz2 = {}
    for c in range(15, 31):
        x = int(round(B_X(c)))
        yc = _disk_center_y(strip(B_BLUE, x, B_YTOP, B_YBOT, 8, 62), B_YTOP)
        if yc is None:
            yc = _disk_center_y(strip(B_BLUE, x, B_YTOP, B_YBOT, 13, 62), B_YTOP)
        b_Mz2[c] = None if yc is None else round(float(B_MZ2(yc)), 4)

    # (c) fitted amplitude at N_q = 51 (dot centre = midpoint of its widest rows)
    pur = np.sqrt(((im[C_YTOP:C_YBOT, C_X51 - 16:C_X51 + 17] - C_PURPLE) ** 2)
                  .sum(axis=2)) < 70
    rc = pur.sum(axis=1)
    rows = np.where(rc >= 0.7 * rc.max())[0]
    amp51 = round(float(C_AMP((rows.min() + rows.max()) / 2.0 + C_YTOP)), 4)

    return {"a_M": a_M, "b_Mz2": b_Mz2, "amplitude": amp51}


def write_csv(data: dict, path: Path) -> None:
    """Write the digitized 51q trace: cycle, M (panel a), Mz2 (panel b)."""
    cycles = list(range(0, 31))
    lines = [
        "# 51-qubit error-mitigated experiment, DIGITIZED from arXiv:2607.24937v1 Fig. 3.",
        "# Approximate (pixel-level, +/- few 1e-3). M: panel (a) QESEM-Extrapolated;",
        f"# Mz2: panel (b) '51q exp.'. Fitted amplitude at N=51 (panel c) = {data['amplitude']}.",
        "cycle,M,Mz2",
    ]
    a, b = data["a_M"], data["b_Mz2"]
    for c in cycles:
        m = 1.0 if c == 0 else a.get(c)       # cycle 0 is the trivial initial state
        v = b.get(c)
        lines.append(f"{c},{'' if m is None else m},{'' if v is None else v}")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "paper_fig3_51q.csv"
    data = digitize(render_page())
    write_csv(data, out)
    occ_a = [c for c, v in data["a_M"].items() if v is None]
    occ_b = [c for c, v in data["b_Mz2"].items() if v is None]
    print(f"wrote {out}")
    print(f"  panel-a M: {31 - 1 - len(occ_a)}/30 cycles"
          + (f" (occluded: {occ_a})" if occ_a else ""))
    print(f"  panel-b Mz2: {16 - len(occ_b)}/16 cycles"
          + (f" (occluded: {occ_b})" if occ_b else ""))
    print(f"  panel-c amplitude(N=51) = {data['amplitude']}")


if __name__ == "__main__":
    main()
