"""The two-point analysis of a KiDS-like survey: masked band powers and a Fisher forecast.

Question: (1) on a patch with star holes and an edge, does the pseudo-C_ell recipe of part 8b
(measure the masked spectrum, subtract the noise, undo the mode coupling) return the convergence
spectrum without bias, and is the B mode it leaves consistent with zero?  (2) With Gaussian
band-power errors, how well does the spectrum of a 1000 deg^2 KiDS-like survey measure
(Omega_m, sigma_8), how long is the banana along S_8, and what is sigma(S_8)?

Computes:
  * the coupling matrices of the masked patch by Monte Carlo (E-only fields with power in one
    band at a time -> masked -> Kaiser-Squires -> pseudo band powers in E and B);
  * the noise bias of the masked patch by Monte Carlo (pure shape noise);
  * 200 lognormal + noise realisations: raw, f_obs-rescaled and decoupled band powers vs truth;
  * the Gaussian Fisher matrix of (Omega_m, sigma_8) from the Limber spectra of w01, for
    l_max = 1500, 3000, 5000, and the likelihood on a grid from a power-law emulator built from
    the same log-derivatives (shows the curved banana).

Writes: figures/ch12/w_pcl.pdf, figures/ch12/w_fisher_cl.pdf, results/ch12/w03_twopoint.tex
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from common import setup, savefig, save_numbers, rng_for, SERIES, DATA, theory_line
import lib_wl as wl

# the 8 analysis bands plus one band below and one above them: power outside the analysed range
# also leaks into it through the mask, so the coupling matrix must cover every multipole
EDGES = np.concatenate([[1.0], wl.CL_EDGES, [1e6]])
IN = slice(1, -1)


def pseudo_eb(g1, g2, mask):
    e, b = wl.kaiser_squires(g1 * mask, g2 * mask)
    return wl.band_power(e, EDGES), wl.band_power(b, EDGES)


def coupling(mask, rng, P, nsim=100):
    """M[X, Y][b, b'] = pseudo power in bin b of X (E or B) when band b' of Y carries the fiducial
    spectrum P with amplitude one (so the shape of C_ell inside each band is the fiducial one)."""
    nb = len(EDGES) - 1
    MEE = np.zeros((nb, nb))
    MBE = np.zeros((nb, nb))
    for j in range(nb):
        Pj = P * ((wl.LMAG >= EDGES[j]) & (wl.LMAG < EDGES[j + 1]))
        for _ in range(nsim):
            k = wl.gaussian_map(Pj, rng.standard_normal((wl.N, wl.N)))
            g1, g2 = wl.shear_from_kappa(k)
            e, b = pseudo_eb(g1, g2, mask)
            MEE[:, j] += e / nsim
            MBE[:, j] += b / nsim
    # E <-> B by a 45-degree rotation of every stick: M_BB = M_EE and M_EB = M_BE
    return np.block([[MEE, MBE], [MBE, MEE]]), MEE, MBE


def coupling_exact(mask, P):
    """The same matrices from the exact expectation (flat-sky MASTER):
    <p_E(l)> = sum_l' w2(l - l') cos^2[2(phi_l - phi_l')] P(l'),  w2 = |FFT(W)|^2 / N_pix^2,
    and sin^2 for B; cos^2 = (1 + cos 4 dphi)/2 turns each sum into two FFT convolutions."""
    npix = mask.size
    w2 = np.abs(np.fft.fft2(mask.astype(float))) ** 2 / npix**2
    W2x = np.fft.ifft2(w2)
    e4 = wl.E2IPHI**2

    def conv(g):                                   # sum_l' w2(l - l') g(l') on the periodic grid
        return npix * np.fft.fft2(W2x * np.fft.ifft2(g))

    nb = len(EDGES) - 1
    idx = np.digitize(wl.LMAG, EDGES) - 1
    MEE = np.zeros((nb, nb))
    MBE = np.zeros((nb, nb))
    for j in range(nb):
        Pj = P * (idx == j)
        c0 = conv(Pj).real
        c4 = (e4 * conv(Pj * np.conj(e4))).real
        pE, pB = 0.5 * (c0 + c4), 0.5 * (c0 - c4)
        for b in range(nb):
            MEE[b, j] = pE[idx == b].mean()
            MBE[b, j] = pB[idx == b].mean()
    return np.block([[MEE, MBE], [MBE, MEE]]), MEE, MBE


def fisher(dC, C, N, edges, area_deg2):
    """Gaussian band-power Fisher matrix: F_ij = sum_b n_modes,b/2 dC_i dC_j / (C+N)^2,
    with n_modes,b = 2 pi l dl * A / (2 pi)^2 summed over the bin (mode counting on the sky)."""
    A = area_deg2 * (np.pi / 180) ** 2
    F = np.zeros((2, 2))
    for lo, hi, in zip(edges[:-1], edges[1:]):
        ell = np.arange(int(np.ceil(lo)), int(np.floor(hi)) + 1)
        nm = ell * A / (2 * np.pi)          # modes per unit l: 2 pi l A /(2 pi)^2, both halves
        w = nm / 2 / (C(ell) + N) ** 2
        d = [f(ell) for f in dC]
        for i in range(2):
            for j in range(2):
                F[i, j] += np.sum(w * d[i] * d[j])
    return F


def main():
    setup()
    sp = dict(np.load(DATA / "ch12" / "w_cl.npz"))
    ells = sp["ells"]
    rng = rng_for("ch12", "w03_twopoint")

    # ================================================================ (1) masked band powers
    mask = wl.make_mask(rng_for("ch12", "w02_ks"))            # the same mask as in w02
    fobs = mask.mean()
    P = wl.cl_on_grid(ells, sp["cl_fid"])
    Mmc, MEEmc, MBEmc = coupling(mask, rng, P)
    M, MEE, MBE = coupling_exact(mask, P)
    mc_dev = np.abs(MEEmc - MEE)[np.abs(MEE) > 0.05 * np.diag(MEE)[None, :]].max() / np.diag(MEE).min()
    mc_dev_diag = np.max(np.abs(np.diag(MEEmc) / np.diag(MEE) - 1))
    print("MC vs exact coupling: max abs dev / min diag", mc_dev, "diag rel", mc_dev_diag)
    lam = float(sp["lam_fid"])
    PG, _ = wl.lognormal_spectrum(P, lam)
    s = wl.sigma_pix()
    nb = len(EDGES) - 1
    # noise bias of the masked patch (in practice: rotate every galaxy at random, Hikage et al.)
    # exact for uniform white noise: <p_E> = <p_B> = N f_obs in every band; a Monte Carlo checks it
    Nn = wl.SIGMA_E**2 / (wl.NGAL / wl.ARCMIN**2)
    noiseE = noiseB = np.full(nb, Nn * fobs)
    mcE = np.zeros(nb)
    for _ in range(400):
        n1, n2 = s * rng.standard_normal((2, wl.N, wl.N))
        e, b = pseudo_eb(n1, n2, mask)
        mcE += (e + b) / 800
    noise_mc_dev = np.max(np.abs(mcE[IN] / (Nn * fobs) - 1))
    truth = np.array([P[(wl.LMAG >= a) & (wl.LMAG < b)].mean() for a, b in zip(EDGES[:-1], EDGES[1:])])
    truth[-1] = P[wl.LMAG >= EDGES[-2]].mean()
    raw, resc, dec, decB, full = [], [], [], [], []
    Minv = np.linalg.inv(M)
    for _ in range(200):
        k = wl.lognormal_map(PG, lam, rng.standard_normal((wl.N, wl.N)))
        g1, g2 = wl.shear_from_kappa(k)
        n1, n2 = s * rng.standard_normal((2, wl.N, wl.N))
        e, b = pseudo_eb(g1 + n1, g2 + n2, mask)
        raw.append(e - noiseE)
        resc.append((e - noiseE) / fobs)
        x = Minv @ np.concatenate([e - noiseE, b - noiseB])
        dec.append(x[:nb] * truth); decB.append(x[nb:] * truth)    # amplitudes -> band powers
        ef, _ = wl.kaiser_squires(g1 + n1, g2 + n2)
        full.append(wl.band_power(ef, EDGES) - wl.SIGMA_E**2 / (wl.NGAL / wl.ARCMIN**2))
    raw, resc, dec, decB, full = (np.array(x)[:, IN] for x in (raw, resc, dec, decB, full))
    truth = truth[IN]
    lb = wl.ell_mean(EDGES)[IN]
    bias_dec = dec.mean(0) / truth - 1
    err_dec = dec.std(0) / np.sqrt(len(dec)) / truth
    bias_resc = resc.mean(0) / truth - 1
    bias_raw = raw.mean(0) / truth - 1
    chiB = np.sum((decB.mean(0) / (decB.std(0) / np.sqrt(len(decB)))) ** 2)
    sd_ratio = dec.std(0) / full.std(0)

    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.2), constrained_layout=True)
    ax[0].imshow((MEE / np.diag(MEE)[None, :])[IN, IN], origin="upper", cmap="Blues", vmin=0, vmax=1)
    ax[0].set_title(r"$M^{EE}_{bb'}$ (columns scaled to diagonal 1)", fontsize=9)
    ax[0].set_xlabel(r"true band $b'$"); ax[0].set_ylabel(r"pseudo band $b$")
    ax[0].grid(False)
    ax[1].plot(lb, bias_raw * 100, "s", mfc="none", color=SERIES[0], label="masked, noise subtracted")
    ax[1].plot(lb, bias_resc * 100, "^", color=SERIES[2], label=r"... divided by $f_{\rm obs}$")
    ax[1].errorbar(lb, bias_dec * 100, yerr=err_dec * 100, fmt="o", color=SERIES[1], label="decoupled (MASTER)")
    ax[1].axhline(0, color="k", lw=0.8, ls="--")
    ax[1].set_xscale("log")
    ax[1].set_xlabel(r"$\ell$"); ax[1].set_ylabel("bias of the band power [%]")
    ax[1].set_xticks([100, 300, 1000, 3000]); ax[1].set_xticklabels(["100", "300", "1000", "3000"])
    ax[1].minorticks_off()
    ax[1].set_ylim(-17, 12)
    ax[1].legend(fontsize=7.5, loc="upper right")
    savefig(fig, "ch12", "w_pcl")

    # ================================================================ (2) Gaussian Fisher forecast
    th = np.array([wl.OM_FID, wl.S8_FID])
    h = wl.STEP * th
    lnC = {k: np.log(sp[f"cl_{k}"]) for k in ("fid", "om_p", "om_m", "s8_p", "s8_m")}

    def interp(arr):
        return lambda l: np.exp(np.interp(np.log(l), np.log(ells), arr))

    C = interp(lnC["fid"])
    dC_om = lambda l: (interp(lnC["om_p"])(l) - interp(lnC["om_m"])(l)) / (2 * h[0])
    dC_s8 = lambda l: (interp(lnC["s8_p"])(l) - interp(lnC["s8_m"])(l)) / (2 * h[1])
    Nn = wl.SIGMA_E**2 / (wl.NGAL / wl.ARCMIN**2)
    out = {}
    for lmax in (1500, 3000, 5000):
        edges = np.logspace(np.log10(100), np.log10(lmax), 13)
        F = fisher([dC_om, dC_s8], C, Nn, edges, wl.AREA_DEG2)
        cov = np.linalg.inv(F)
        # S8 = sigma8 (Om/0.3)^0.5: gradient
        S8 = th[1] * np.sqrt(th[0] / 0.3)
        g = np.array([0.5 * S8 / th[0], S8 / th[1]])
        sS8 = np.sqrt(g @ cov @ g)
        r = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        # the best-measured direction: sigma8 Om^alpha, alpha from the major axis of the ellipse in logs
        Jl = np.diag(th)                                    # d theta / d ln theta
        Fl = Jl @ F @ Jl
        covl = np.linalg.inv(Fl)
        w, v = np.linalg.eigh(covl)
        major = v[:, np.argmax(w)]                          # long axis in (ln Om, ln s8)
        alpha = -major[1] / major[0]
        fom = 1 / np.sqrt(np.linalg.det(cov))
        out[lmax] = dict(F=F, cov=cov, sS8=sS8, r=r, alpha=alpha, fom=fom)
        print(lmax, "sig Om", np.sqrt(cov[0, 0]), "sig s8", np.sqrt(cov[1, 1]), "r", r, "sS8", sS8,
              "alpha", alpha, "fom", fom, "fixed-Om sig s8", 1 / np.sqrt(F[1, 1]))

    # likelihood on a grid with the power-law emulator ln C = ln C_fid + a ln(s8/s8f) + b ln(Om/Omf)
    lmax = 3000
    edges = np.logspace(np.log10(100), np.log10(lmax), 13)
    lc = np.sqrt(edges[1:] * edges[:-1])
    a = (np.interp(np.log(lc), np.log(ells), lnC["s8_p"]) - np.interp(np.log(lc), np.log(ells), lnC["s8_m"])) / (
        np.log(1 + wl.STEP) - np.log(1 - wl.STEP))
    b = (np.interp(np.log(lc), np.log(ells), lnC["om_p"]) - np.interp(np.log(lc), np.log(ells), lnC["om_m"])) / (
        np.log(1 + wl.STEP) - np.log(1 - wl.STEP))
    A = wl.AREA_DEG2 * (np.pi / 180) ** 2
    nm = np.array([np.sum(np.arange(int(np.ceil(lo)), int(np.floor(hi)) + 1)) * A / (2 * np.pi)
                   for lo, hi in zip(edges[:-1], edges[1:])])
    dl = edges[1:] - edges[:-1]
    Cf = C(lc)
    var = 2 * (Cf + Nn) ** 2 / nm                           # variance of a band average
    oms = np.linspace(0.12, 0.65, 300)
    s8s = np.linspace(0.45, 1.35, 300)
    OM, SG = np.meshgrid(oms, s8s, indexing="ij")
    model = Cf[None, None, :] * np.exp(a * np.log(SG[..., None] / th[1]) + b * np.log(OM[..., None] / th[0]))
    chi2 = np.sum((model - Cf) ** 2 / var, axis=-1)

    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    ax.contour(OM, SG, chi2, levels=[2.30, 6.18], colors=[SERIES[0], SERIES[0]], linestyles=["-", "--"])
    for k, (lm, c) in enumerate(((3000, SERIES[1]),)):
        cov = out[lm]["cov"]
        w, v = np.linalg.eigh(cov)
        ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
        for nsig, ls in ((2.30, "-"), (6.18, "--")):
            e = Ellipse(th, 2 * np.sqrt(nsig * w[1]), 2 * np.sqrt(nsig * w[0]), angle=ang, fill=False,
                        color=c, ls=ls, lw=1.2)
            ax.add_patch(e)
    S8f = th[1] * np.sqrt(th[0] / 0.3)
    xx = np.linspace(0.12, 0.65, 200)
    ax.plot(xx, S8f * np.sqrt(0.3 / xx), color="k", lw=0.8, ls=":")
    ax.plot(*th, "k+", ms=8)
    ax.set_xlim(0.12, 0.65); ax.set_ylim(0.45, 1.35)
    ax.set_xlabel(r"$\Omega_m$"); ax.set_ylabel(r"$\sigma_8$")
    ax.plot([], [], color=SERIES[0], label="likelihood (power-law emulator)")
    ax.plot([], [], color=SERIES[1], label="Fisher ellipse")
    ax.plot([], [], color="k", ls=":", label=r"constant $S_8$")
    ax.legend(fontsize=8, loc="upper right")
    savefig(fig, "ch12", "w_fisher_cl")
    # S8 from the grid likelihood (marginal over the 2-D posterior with a flat prior)
    post = np.exp(-0.5 * (chi2 - chi2.min()))
    S8g = SG * np.sqrt(OM / 0.3)
    hist, be = np.histogram(S8g.ravel(), bins=400, weights=post.ravel())
    cen = 0.5 * (be[1:] + be[:-1])
    m1 = np.sum(cen * hist) / hist.sum()
    sS8_grid = np.sqrt(np.sum((cen - m1) ** 2 * hist) / hist.sum())
    pom = post.sum(1)
    mom = np.sum(oms * pom) / pom.sum()
    sOm_grid = np.sqrt(np.sum((oms - mom) ** 2 * pom) / pom.sum())

    o3 = out[3000]
    save_numbers("ch12", "w03_twopoint", {
        "WFobsT": f"{fobs:.3f}",
        "WBiasRawMin": f"{100 * bias_raw.min():.0f}", "WBiasRawMax": f"{100 * bias_raw.max():.0f}",
        "WBiasRescMax": f"{100 * np.abs(bias_resc).max():.0f}",
        "WBiasDecMax": f"{100 * np.abs(bias_dec).max():.1f}",
        "WErrDecMax": f"{100 * err_dec.max():.1f}",
        "WChiB": f"{chiB:.1f}", "WNbands": nb - 2,
        "WSdRatio": f"{np.median(sd_ratio):.2f}",
        "WLeakMax": f"{100 * np.max((MBE / np.diag(MEE)[None, :])[IN, IN]):.1f}",
        "WOffDiag": f"{100 * np.max(((MEE - np.diag(np.diag(MEE))) / np.diag(MEE)[None, :])[IN, IN]):.0f}",
        "WSigOm": f"{np.sqrt(o3['cov'][0, 0]):.3f}", "WSigSe": f"{np.sqrt(o3['cov'][1, 1]):.3f}",
        "WRcorr": f"{o3['r']:.3f}", "WSigSEight": f"{o3['sS8']:.4f}",
        "WSigSEightRel": f"{100 * o3['sS8'] / S8f:.1f}",
        "WAlphaF": f"{o3['alpha']:.2f}", "WFomCl": f"{o3['fom']:.0f}",
        "WSigSeFixed": f"{1 / np.sqrt(o3['F'][1, 1]):.4f}",
        "WSigSEightLow": f"{out[1500]['sS8']:.4f}", "WSigSEightHigh": f"{out[5000]['sS8']:.4f}",
        "WSigSEightGrid": f"{sS8_grid:.4f}", "WSigOmGrid": f"{sOm_grid:.3f}",
        "WMargFactor": f"{1 / np.sqrt(1 - o3['r']**2):.1f}",
        "WNoiseN": "%.2f\\times10^{%d}" % (Nn / 10**np.floor(np.log10(Nn)), np.floor(np.log10(Nn))),
        "WMcDiag": f"{100 * mc_dev_diag:.1f}",
        "WNoiseMcDev": f"{100 * noise_mc_dev:.2f}",
        "WChiE": f"{np.sum(((dec.mean(0) - truth) / (dec.std(0) / np.sqrt(len(dec)))) ** 2):.1f}",
    })
    print("bias raw", bias_raw, "resc", bias_resc, "dec", bias_dec, "err", err_dec, "chiB", chiB)
    print("sd ratio", sd_ratio, "grid", sS8_grid, sOm_grid)


if __name__ == "__main__":
    main()
