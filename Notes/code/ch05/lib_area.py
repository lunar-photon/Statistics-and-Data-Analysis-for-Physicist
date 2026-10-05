"""lib_area.py -- Bayes' theorem as information shrinking the sample space.

Two pictures of the same areas, for any set of hypotheses H_i and evidence E:

* the sample space has area 1 and is cut into strips, strip i as wide as the
  prior P(H_i);
* the part of strip i where E is true has area P(H_i) P(E | H_i), the joint
  probability; once E is observed only those parts survive, and the posterior
  P(H_i | E) is the share of the surviving area that lies in strip i.

box_diagram draws E as one box: in strip i a full-height piece covering the
fraction P(E | H_i) of the strip's width.  area_diagram draws the unit square with
bars: in strip i a bar of height P(E | H_i), everything above the bars greyed out.

Functions
---------
box_pieces(priors, likes)                  the strips and the pieces of E (geometry)
box_diagram(ax, priors, likes, ...)        the box picture, before or after the evidence
area_diagram(ax, priors, likes, ...)       the unit square with bars
fraction_panel(ax, priors, likes, k, ...)  posterior of hypothesis k drawn as
                                           [rectangle] / ([rectangle] + [rectangle] + ...)
posterior(priors, likes)                   the numbers themselves
"""
import numpy as np
import matplotlib.patches as mpatches

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import SERIES, INK, INK2

GREY = "#5b5b5b"      # the discarded region: E is false there
STRIP = "#f1f0ec"     # a hypothesis strip before we know anything about E


def posterior(priors, likes):
    """Return (joint areas, P(E), posteriors) for a partition of hypotheses."""
    priors, likes = np.asarray(priors, float), np.asarray(likes, float)
    joint = priors * likes                 # area of each surviving rectangle
    pE = joint.sum()                       # total surviving area = P(E), the sum over paths
    return joint, pE, joint / pE


def area_diagram(ax, priors, likes, labels=None, after=True, colors=None,
                 brace_labels=True, title=None, fontsize=9, xlim=(0, 1)):
    """Draw the unit square split at the priors, with likelihood bars.

    after=False : the square before the evidence (strips and bars visible)
    after=True  : the region where E is false is greyed out
    xlim        : lets an inset zoom into a thin strip (e.g. a 1% prior)
    """
    priors, likes = np.asarray(priors, float), np.asarray(likes, float)
    k = len(priors)
    colors = colors or SERIES[:k]
    labels = labels or [f"$H_{i+1}$" for i in range(k)]
    edges = np.concatenate([[0.0], np.cumsum(priors)])
    for i in range(k):
        x0, w = edges[i], priors[i]
        # the strip of hypothesis i: its width is the prior
        ax.add_patch(mpatches.Rectangle((x0, 0), w, 1, fc=STRIP, ec="none"))
        # the part of the strip where E is true: height = likelihood, area = joint
        ax.add_patch(mpatches.Rectangle((x0, 0), w, likes[i], fc=colors[i], ec="none"))
        if after:   # the part where E is false is discarded
            ax.add_patch(mpatches.Rectangle((x0, likes[i]), w, 1 - likes[i],
                                            fc=GREY, ec="none", alpha=0.85))
        if i > 0:
            ax.plot([x0, x0], [0, 1], color=INK2, lw=0.8)
    ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, fc="none", ec=INK, lw=1.0))
    if brace_labels:
        for i in range(k):
            xc = edges[i] + priors[i] / 2
            if xlim[0] <= xc <= xlim[1]:
                ax.text(xc, -0.04, labels[i], ha="center", va="top", fontsize=fontsize)
    ax.set_xlim(xlim[0] - 0.02 * (xlim[1] - xlim[0]), xlim[1] + 0.02 * (xlim[1] - xlim[0]))
    ax.set_ylim(-0.14, 1.03)
    ax.set_aspect("auto")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=fontsize + 1)
    return edges


def fraction_panel(ax, priors, likes, k=0, colors=None, scale=1.0, fontsize=10):
    """Draw P(H_k | E) as the rectangle of H_k over the sum of all surviving rectangles.

    Every rectangle keeps its true width (prior) and height (likelihood), times `scale`.
    """
    priors, likes = np.asarray(priors, float), np.asarray(likes, float)
    n = len(priors)
    colors = colors or SERIES[:n]
    joint, pE, post = posterior(priors, likes)
    gap = 0.12 * scale
    widths = priors * scale
    heights = likes * scale
    # denominator: every surviving rectangle (zero-area ones are dropped), joined by '+'
    keep = [i for i in range(n) if joint[i] > 0]
    den_w = widths[keep].sum() + gap * (len(keep) - 1)
    x = 0.0
    top = -0.05 * scale
    for j, i in enumerate(keep):
        ax.add_patch(mpatches.Rectangle((x, top - heights[i]), widths[i], heights[i],
                                        fc=colors[i], ec=colors[i], lw=0.8))
        if j < len(keep) - 1:
            ax.text(x + widths[i] + gap / 2, top - 0.5 * heights[keep].max(), "+",
                    ha="center", va="center", fontsize=fontsize)
        x += widths[i] + gap
    # numerator, centred over the fraction bar
    ax.add_patch(mpatches.Rectangle(((den_w - widths[k]) / 2, 0.05 * scale),
                                    widths[k], heights[k], fc=colors[k], ec=colors[k], lw=0.8))
    ax.plot([-0.03 * scale, den_w + 0.03 * scale], [0, 0], color=INK, lw=1.2)
    ax.text(den_w + 0.08 * scale, 0, rf"$=\;{joint[k]:.4g}\,/\,{pE:.4g}\;=\;{post[k]:.3f}$",
            ha="left", va="center", fontsize=fontsize)
    ax.set_xlim(-0.05 * scale, den_w + 0.9 * scale)
    ax.set_ylim(top - heights[keep].max() - 0.05 * scale, 0.1 * scale + heights[k])
    ax.set_aspect("equal")
    ax.axis("off")
    return post


# ---------------------------------------------------------------------------
# The box picture: the evidence E drawn as one box inside the sample space.
# ---------------------------------------------------------------------------
def _tint(hex_colour, a):
    """The colour hex_colour laid at opacity a on white (as TikZ's colour!a)."""
    c = np.array([int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    return tuple(a * c + (1 - a))


BLUE_TINT = _tint("#2E6FBF", 0.18)    # a hypothesis strip (formblue!18)
GREEN = _tint("#1E8449", 0.45)        # where E is true inside a strip (tangentgreen!45)
ORANGE = "#D35400"                    # the outline of E (fibreorange)
ORANGE_TXT = "#A94300"                # its label (fibreorange!80!black)
W_OMEGA = 1.5                         # Omega is 1.5 x 1 (shown with area 1)


def box_pieces(priors, likes):
    """Strip edges and the green piece [a, b] of each strip, in Omega's coordinates.

    Strip i is as wide as its prior; its piece runs the full height and covers the
    fraction likes[i] of the strip's width, so its area is prior x likelihood.
    Pieces are pushed against the neighbouring strip that also has a piece, so the
    pieces of E join into as few boxes as possible (one box for two hypotheses).
    """
    priors, likes = np.asarray(priors, float), np.asarray(likes, float)
    k = len(priors)
    edges = W_OMEGA * np.concatenate([[0.0], np.cumsum(priors)])
    pieces = []
    for i in range(k):
        x0, x1 = edges[i], edges[i + 1]
        w = (x1 - x0) * likes[i]
        left_has = i > 0 and likes[i - 1] > 0
        right_has = i < k - 1 and likes[i + 1] > 0
        if right_has and not left_has:     # lean right, towards the next piece
            pieces.append((x1 - w, x1))
        else:                              # lean left (or fill the strip if likes[i] = 1)
            pieces.append((x0, x0 + w))
    return edges, pieces


def _boxes(pieces):
    """Join touching pieces into the boxes that make up E."""
    out = []
    for a, b in pieces:
        if b - a <= 0:
            continue
        if out and abs(out[-1][1] - a) < 1e-12:
            out[-1][1] = b
        else:
            out.append([a, b])
    return out


def _shapes(ax, edges, pieces, after):
    """Draw Omega, the strips, the green pieces and the orange outline of E."""
    if not after:      # before the evidence: the whole of Omega, strip by strip
        ax.add_patch(mpatches.Rectangle((0, 0), W_OMEGA, 1, fc=BLUE_TINT, ec="none"))
    for a, b in pieces:  # the part of each strip where E is true
        if b > a:
            ax.add_patch(mpatches.Rectangle((a, 0), b - a, 1, fc=GREEN, ec="none"))
    for x in edges[1:-1]:  # the lines between strips
        ax.plot([x, x], [0, 1], color=INK if not after else "#9a9a9a",
                lw=0.6, ls="-" if not after else "--", zorder=2)
    ax.add_patch(mpatches.Rectangle((0, 0), W_OMEGA, 1, fc="none",
                                    ec=INK if not after else "#9a9a9a",
                                    lw=1.0 if not after else 0.7,
                                    ls="-" if not after else "--"))
    for a, b in _boxes(pieces):   # E itself: outside it is ruled out once E is seen
        ax.add_patch(mpatches.Rectangle((a, 0), b - a, 1, fc="none", ec=ORANGE,
                                        lw=1.8, zorder=4))


def box_diagram(ax, priors, likes, labels=None, evidence_label="$E$", after=False,
                inset=None, fontsize=8, title=None, numbers=True, show_priors=True):
    """The box picture of conditioning, drawn to scale.

    Omega has area 1 and is cut into strips as wide as the priors; E (orange
    outline) covers the fraction likes[i] of strip i, so each green piece has area
    prior x likelihood.
    after=False : Omega before the evidence, with the areas of the pieces written in
    after=True  : everything outside E is dashed out; each hypothesis is labelled with
                  the share of E its piece fills, the posterior
    inset       : None, or dict(x=(x0, x1) or x="box", bounds=[left, bottom, width, height]) to
                  magnify the probability range x0..x1 (full height) in an inset placed
                  at `bounds` (axes fraction), so that a thin strip can be seen
    Returns the posteriors.
    """
    priors, likes = np.asarray(priors, float), np.asarray(likes, float)
    k = len(priors)
    labels = labels or [f"$H_{i+1}$" for i in range(k)]
    joint = priors * likes
    pE = joint.sum()
    post = joint / pE if pE > 0 else np.full(k, np.nan)
    edges, pieces = box_pieces(priors, likes)

    if inset is None:
        win = None
    elif isinstance(inset["x"], str) and inset["x"] == "box":
        # zoom window = exactly the evidence box (no white margins inside the zoom)
        used = [(a, b) for (a, b) in pieces if b > a]
        win = (min(a for a, _ in used), max(b for _, b in used))
    else:
        win = (W_OMEGA * inset["x"][0], W_OMEGA * inset["x"][1])
    ylab = 0.5 if inset is None else 0.08     # keep strip labels clear of an inset (and of its labels)

    def annotate(a, xlim, small):
        """Labels for axes `a` showing Omega's x-range `xlim`; `small` = inset."""
        span = xlim[1] - xlim[0]
        fs = fontsize - (1 if small else 0)
        for i in range(k):
            pa, pb = pieces[i]
            x0, x1 = edges[i], edges[i + 1]
            # the hypothesis label: in the free part of its strip (before) or in its piece
            free = [(x0, pa), (pb, x1)]
            fa, fb = max(free, key=lambda t: t[1] - t[0])
            if not after and (fb - fa) / span > 0.10 and xlim[0] <= (fa + fb) / 2 <= xlim[1]:
                xl = (fa + fb) / 2 if (small or win is None) else fa + 0.88 * (fb - fa)
                a.text(xl, 0.5 if small else ylab, labels[i], ha="center",
                       va="center", fontsize=fs)
            elif not after and not small and show_priors and (x1 - x0) < 0.25:
                # a strip too thin to hold its label: name it under Omega, with its prior
                a.text(x0, -0.08, f"{labels[i]}: {priors[i]:.3g}", ha="left", va="top",
                       fontsize=fs)
            if not numbers or pb <= pa:
                if after and pb <= pa and (x1 - x0) / span > 0.12 and xlim[0] <= (x0 + x1) / 2 <= xlim[1]:
                    a.text((x0 + x1) / 2, 0.5, labels[i] + "\nruled out", ha="center",
                           va="center", fontsize=fs, color="#7F8C8D")
                continue
            val = f"{post[i]:.3g}" if after else f"{joint[i]:.3g}"
            txt = (labels[i] + "\n" + val) if (after or small) else val
            xc = (pa + pb) / 2
            if not (xlim[0] <= xc <= xlim[1]):
                continue
            if not small and win is not None and win[0] <= xc <= win[1]:
                continue                       # this piece is labelled in the inset
            if (pb - pa) / span > 0.13:        # wide enough: write it inside the piece
                a.text(xc, 0.5 if after else 0.3, txt, ha="center", va="center", fontsize=fs)
            elif small:                        # too thin inside an inset: label just below it
                a.annotate(txt.replace("\n", ": "), xy=(xc, 0.25), xytext=(xc + 0.06 * span, -0.10),
                           fontsize=fs, va="top", ha="left", annotation_clip=False,
                           arrowprops=dict(arrowstyle="-", color="#7F8C8D", lw=0.5))
            elif not small:                    # too thin: a leader line to the right
                ytxt = 0.78 if i % 2 == 0 else 0.36
                right = max(b for a_, b in _boxes(pieces) if a_ <= pa + 1e-12)  # leave the box
                a.annotate(txt, xy=(xc, ytxt), xytext=(min(right, xlim[1]) + 0.07 * span, ytxt),
                           fontsize=fs, va="center", ha="left",
                           arrowprops=dict(arrowstyle="-", color="#7F8C8D", lw=0.5))

    _shapes(ax, edges, pieces, after)
    annotate(ax, (0, W_OMEGA), False)
    boxes = _boxes(pieces)
    if boxes:   # the label of E above its (first) box
        a, b = boxes[0][0], boxes[-1][1]
        lab = evidence_label if not after else f"{evidence_label}, area {pE:.3g}"
        ax.text((a + b) / 2 if (b - a) > 0.3 else a, 1.03, lab, ha="center" if (b - a) > 0.3 else "left",
                va="bottom", fontsize=fontsize, color=ORANGE_TXT)
    if not after and show_priors:   # the priors under the strips
        for i in range(k):
            x0, x1 = edges[i], edges[i + 1]
            if (x1 - x0) >= 0.25:
                ax.plot([x0 + 0.01, x0 + 0.01, x1 - 0.01, x1 - 0.01], [-0.03, -0.06, -0.06, -0.03],
                        color="#7F8C8D", lw=0.5)
                ax.text((x0 + x1) / 2, -0.08, f"{priors[i]:.3g}", ha="center", va="top",
                        fontsize=fontsize)
    ax.set_xlim(-0.02, W_OMEGA + 0.02)
    ax.set_ylim(-0.2, 1.14)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=fontsize + 1)

    if inset is not None:   # magnify a thin strip horizontally
        lo, hi = win
        axin = ax.inset_axes(inset["bounds"])
        _shapes(axin, edges, pieces, after)
        annotate(axin, (lo, hi), True)
        axin.set_xlim(lo, hi)
        axin.set_ylim(0, 1)
        axin.set_aspect("auto")
        axin.set_xticks([]); axin.set_yticks([]); axin.grid(False)
        for s in axin.spines.values():
            s.set_visible(True); s.set_color(INK2); s.set_linewidth(0.6)
        ax.add_patch(mpatches.Rectangle((lo, 0), hi - lo, 1, fc="none", ec=INK2, lw=0.8, zorder=5))
        # zoom lines: each corner of the window joins the MATCHING corner of the inset.
        # ConnectionPatch anchors one end in ax's data coordinates and the other in the inset's
        # axes fraction, so the lines stay on the corners whatever the final figure layout is.
        from matplotlib.patches import ConnectionPatch
        for cx in (0, 1):
            for cy in (0, 1):
                con = ConnectionPatch(xyA=((lo, hi)[cx], cy), coordsA="data", axesA=ax,
                                      xyB=(cx, cy), coordsB="axes fraction", axesB=axin,
                                      color="#9a9a96", lw=0.7, zorder=4)
                ax.add_artist(con)
        axin.set_zorder(6)
        mag = W_OMEGA * inset["bounds"][2] / (hi - lo) * (ax.get_xlim()[1] - ax.get_xlim()[0]) / W_OMEGA
        axin.text(0.5, 1.03, f"$\\times{mag:.0f}$ wide", transform=axin.transAxes,
                  ha="center", va="bottom", fontsize=fontsize - 1, color=INK2)
    return post
