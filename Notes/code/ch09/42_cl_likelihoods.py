"""42_cl_likelihoods.py -- the exact full-sky C_l likelihood and its approximations; Hamimeche & Lewis Fig. 1.

Question: on the full sky, (2l+1) C_hat_l / C_l ~ chi^2_{2l+1}, so the likelihood of a candidate
C_l given the observed C_hat_l is known exactly.  Real analyses use approximations (they extend
to the cut sky).  How do the shapes differ at one multipole, and how wrong is the amplitude a
bin of 10 multipoles prefers under each approximation?
Reproduces Fig. 1 of Hamimeche & Lewis (2008, arXiv:0801.0554): for bins l = lmin..lmin+9, many
simulated full-sky noiseless spectra (C_l = A C_l^fid with A = 1, so only x_l = C_hat_l / C_l^fid
matters), the best-fit amplitude A under
    exact      -2 ln L = sum (2l+1) [x/A + ln A]                     (their eq. 8 / 16)
    S          sum (2l+1)/2 [(x - A)/x]^2                              (22)
    Q          sum (2l+1)/2 [(x - A)/A]^2                              (23)
    D          Q + sum ln A                                            (25, as printed)
    LN         sum (2l+1)/2 [ln(x/A)]^2                                (26)
    WMAP       1/3 Q + 2/3 LN                                          (27)
    cube root  (28) with alpha = -1 (Gaussian in C_hat^(1/3))
and plots |<A_i - A_exact>| and sqrt<(A_i - A_exact)^2> against lmin with the tolerance 1/lmin.
The published curves are read off the vector graphics of the paper's PDF
(references/HamimecheLewis2008.pdf, page 12) with PyMuPDF and overlaid.
Writes: figures/ch09/cl_like_shapes.pdf, figures/ch09/hl_fig1.pdf, results/ch09/42_cl_likelihoods.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

setup()
rng = rng_for("ch09", "42_cl_likelihoods")
nums = {}


# ---------------------------------------------------------------- -2 ln L for one multipole, A = C / C_fid
def m2lnL(kind, x, A, l):
    """-2 ln L (up to a constant) at multipole l, observed x = C_hat / C_fid, candidate C = A C_fid."""
    nu = 2 * l + 1.0
    if kind == "exact":
        return nu * (x / A + np.log(A))
    if kind == "S":
        return nu / 2 * ((x - A) / x) ** 2
    if kind == "Q":
        return nu / 2 * ((x - A) / A) ** 2
    if kind == "D":
        return nu / 2 * ((x - A) / A) ** 2 + np.log(A)
    if kind == "f":                                    # fixed fiducial in the denominator (24)
        return nu / 2 * (x - A) ** 2
    if kind == "LN":
        return nu / 2 * np.log(x / A) ** 2
    if kind == "WMAP":
        return (1 / 3) * m2lnL("Q", x, A, l) + (2 / 3) * m2lnL("LN", x, A, l)
    if kind == "cube":
        al = -1.0
        c = ((2 * l + al) / nu) ** (1 / 3)
        return nu * 4.5 * c * ((x / A) ** (1 / 3) - c) ** 2 + (1 - al) * np.log(A)
    if kind == "HL":                                   # g(x)^2 form, eq. (44); equals exact - const
        z = x / A
        g2 = 2 * (z - np.log(z) - 1)
        return nu / 2 * g2
    raise ValueError(kind)


# ---------------------------------------------------------------- figure 1: shapes at one multipole
fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.3))
Agrid = np.linspace(0.35, 3.0, 600)
kinds = [("exact", "exact $\\chi^2_{2\\ell+1}$", "k", "-"), ("S", "$\\mathcal{L}_S$", SERIES[0], "--"),
         ("Q", "$\\mathcal{L}_Q$", SERIES[1], "--"), ("f", "$\\mathcal{L}_f$ (fiducial)", SERIES[2], "-."),
         ("LN", "log-normal", SERIES[4], ":"), ("WMAP", "WMAP", SERIES[3], "-")]
for ax, l, xobs in ((axs[0], 5, 0.7), (axs[1], 30, 0.85)):
    for kind, lab, col, ls in kinds:
        v = m2lnL(kind, xobs, Agrid, l)
        L = np.exp(-0.5 * (v - v.min()))
        ax.plot(Agrid, L, color=col, ls=ls, lw=1.3 if kind == "exact" else 1.1, label=lab)
    ax.axvline(xobs, color="0.5", lw=0.6, ls=":")
    ax.set_xlabel(r"candidate $C_\ell/C_\ell^{\rm fid}$")
    ax.set_title(f"$\\ell={l}$, observed $\\hat C_\\ell={xobs}\\,C^{{\\rm fid}}_\\ell$", fontsize=9)
    ax.set_xlim(Agrid[0], 2.4 if l == 5 else 1.6)
axs[0].set_ylabel("likelihood (peak 1)")
axs[0].legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch09", "cl_like_shapes")

# where the exact likelihood peaks and what its mean is, at l = 5 (HL below eq. 8)
l5 = 5
nums.update({"NineCLikeMeanFactor": f"{(2 * l5 + 1) / (2 * l5 - 3):.3f}",
             "NineCLikeModeFactor": f"{(2 * l5 - 1) / (2 * l5 + 1):.3f}"})


# ---------------------------------------------------------------- HL Fig. 1: amplitude of a bin of 10
def golden(f, lo, hi, it=80):
    """Vectorised golden-section minimiser: one minimum per row."""
    g = (np.sqrt(5) - 1) / 2
    a, b = lo.copy(), hi.copy()
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(it):
        m = fc < fd
        b = np.where(m, d, b); a = np.where(m, a, c)
        c, d = np.where(m, b - g * (b - a), d), np.where(m, c, a + g * (b - a))
        fc, fd = f(c), f(d)
    return 0.5 * (a + b)


NSIM, DL = 10000, 10
LMINS = np.unique(np.round(np.geomspace(2, 1500, 70)).astype(int))
NAMES = ["S", "Q", "D", "LN", "WMAP", "cube"]
bias = {k: np.empty(LMINS.size) for k in NAMES}
rms = {k: np.empty(LMINS.size) for k in NAMES}
for i, lmin in enumerate(LMINS):
    ls = np.arange(lmin, lmin + DL)
    nu = 2 * ls + 1.0
    x = rng.chisquare(nu[None, :], size=(NSIM, DL)) / nu          # x = C_hat / C_true
    A_ex = (nu * x).sum(1) / nu.sum()                              # exact maximum, HL eq. (17)
    lo, hi = np.full(NSIM, 0.2), np.full(NSIM, 4.0)
    est = {"S": (nu / x).sum(1) / (nu / x ** 2).sum(1),           # closed forms
           "Q": (nu * x ** 2).sum(1) / (nu * x).sum(1),
           "LN": np.exp((nu * np.log(x)).sum(1) / nu.sum())}
    for k in ("D", "WMAP", "cube"):
        est[k] = golden(lambda A, k=k: m2lnL(k, x, A[:, None], ls[None, :]).sum(1), lo, hi)
    for k in NAMES:
        bias[k][i] = abs(np.mean(est[k] - A_ex))
        rms[k][i] = np.sqrt(np.mean((est[k] - A_ex) ** 2))
    # check of the HL form: identical maximum to the exact likelihood
    if lmin == LMINS[len(LMINS) // 2]:
        A_hl = golden(lambda A: m2lnL("HL", x, A[:, None], ls[None, :]).sum(1), lo, hi)
        nums["NineCHLcheck"] = float(f"{np.max(np.abs(A_hl - A_ex)):.1e}")

j = np.argmin(np.abs(LMINS - 1000))
j2 = np.argmin(np.abs(LMINS - 100))
for k, tag in zip(NAMES, ["S", "Q", "D", "LN", "WMAP", "Cube"]):
    nums[f"NineCHLbias{tag}"] = float(f"{bias[k][j]:.2e}")
    nums[f"NineCHLrms{tag}"] = float(f"{rms[k][j]:.2e}")
    nums[f"NineCHLrms{tag}Hund"] = float(f"{rms[k][j2]:.2e}")
nums.update({"NineCHLlref": int(LMINS[j]), "NineCHLlrefHund": int(LMINS[j2]), "NineCHLnsim": NSIM,
             "NineCHLQoverL": f"{bias['Q'][j] * LMINS[j]:.2f}", "NineCHLSoverL": f"{bias['S'][j] * LMINS[j]:.2f}",
             "NineCHLDoverL": f"{bias['D'][j] * LMINS[j]:.2f}"})

# ---------------------------------------------------------------- the published curves, digitised
def digitise_hl():
    """Paths of HL Fig. 1 -> data coordinates.  Axis calibration from the tick marks on the page:
    x: l = 500 at 197.45 pt, l = 1000 at 251.37 pt (left panel; right panel shifted by 169.4 pt);
    y: left panel 10 at 227.13 pt down to 1e-9 at 517.47 pt; right panel 10 down to 1e-7."""
    import pymupdf
    pdf = pathlib.Path(__file__).resolve().parents[2] / "references" / "HamimecheLewis2008.pdf"
    if not pdf.exists():
        return None
    dr = pymupdf.open(pdf)[11].get_drawings()
    ids = {"left": {"tol": [33], "S": [35, 36], "Q": [55, 56], "D": [45, 46], "WMAP": [40, 41], "cube": [50, 51]},
           "right": {"tol": [89], "S": [92], "Q": [120], "D": [106], "WMAP": [99], "cube": [113]}}
    out = {}
    for panel, d in ids.items():
        x0 = 0.0 if panel == "left" else 169.4
        top, bot = 227.13, 517.47
        lo_dec = -9 if panel == "left" else -7
        for k, idx in d.items():
            pts = []
            for i in idx:
                for it in dr[i]["items"]:
                    if it[0] == "l":
                        pts += [(it[1].x, it[1].y), (it[2].x, it[2].y)]
            pts = np.array(pts)
            # drop the short legend samples drawn inside some of the same paths
            lx = (195, 275) if panel == "left" else (364, 450)
            leg = (pts[:, 0] > lx[0]) & (pts[:, 0] < lx[1]) & (pts[:, 1] < 330)
            pts = pts[~leg]
            ell = 500 + (pts[:, 0] - x0 - 197.45) / (251.37 - 197.45) * 500
            logy = 1 - (pts[:, 1] - top) / (bot - top) * (1 - lo_dec)
            o = np.argsort(ell)
            out[(panel, k)] = (ell[o], 10 ** logy[o])
    return out


pub = digitise_hl()
if pub is not None:                                   # the published numbers at l_min = 1000, for the table
    for k, tag in zip(NAMES, ["S", "Q", "D", "LN", "WMAP", "Cube"]):
        if ("right", k) in pub:
            e_, v_ = pub[("right", k)]
            nums[f"NineCHLpubRms{tag}"] = float(f"{np.interp(1000, e_, v_):.1e}")
            e_, v_ = pub[("left", k)]
            if k in ("S", "Q", "D"):
                nums[f"NineCHLpubBias{tag}"] = float(f"{np.interp(1000, e_, v_):.1e}")

fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.8), sharex=True)
sty = {"S": (SERIES[0], ":"), "Q": ("k", "--"), "D": (SERIES[6], "-."), "LN": (SERIES[4], "-"),
       "WMAP": (SERIES[2], "--"), "cube": (SERIES[4], "-.")}
labs = {"S": r"$\mathcal{L}_S$", "Q": r"$\mathcal{L}_Q$", "D": r"Gaussian$_D$", "LN": "log-normal",
        "WMAP": "WMAP", "cube": "cube root (1/3)"}
for ax, d, title in ((axs[0], bias, r"(a) bias $|\langle A_i-A_{\rm exact}\rangle|$"),
                     (axs[1], rms, r"(b) scatter $\langle (A_i-A_{\rm exact})^2\rangle^{1/2}$")):
    ax.loglog(LMINS, 1.0 / LMINS, color=SERIES[7], lw=1.4, label=r"tolerance $1/\ell_{\min}$")
    for k in NAMES:
        if k == "LN" and d is bias:
            pass
        c, s = sty[k]
        ax.loglog(LMINS, d[k], color=c, ls=s, lw=1.1, label=labs[k])
        if pub is not None and k != "LN":
            e_, v_ = pub[("left" if d is bias else "right", k)]
            ax.plot(e_[::3], v_[::3], ".", ms=1.6, color=c, alpha=0.6)
    ax.set_xlabel(r"$\ell_{\min}$ (bin $\ell_{\min}\ldots\ell_{\min}+9$)")
    ax.set_title(title, fontsize=9)
axs[0].set_ylabel("difference in best-fit amplitude")
axs[1].plot([], [], ".", color="0.4", ms=3, label="Hamimeche & Lewis, Fig. 1")
axs[1].legend(fontsize=7, loc="lower left")
fig.tight_layout()
savefig(fig, "ch09", "hl_fig1")
save_numbers("ch09", "42_cl_likelihoods", nums)
print(nums)
