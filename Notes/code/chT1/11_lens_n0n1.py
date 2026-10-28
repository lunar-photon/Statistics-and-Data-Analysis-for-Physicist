"""Lensing reconstruction on simulated flat patches: the quadratic estimator and its N0 and N1 biases.

Question: the quadratic estimator turns pairs of CMB modes into an estimate of the lensing
potential psi. The power spectrum of that estimate is not C^psipsi. What else is in it, how large
is each part, and do the formulas for them agree with simulations?

Computes, on a periodic 17 x 17 deg patch (512^2 pixels of 2 arcmin), temperature only:
  * lensed skies: the unlensed T and the potential psi of CAMB's spectra on a 1024^2 grid of 1 arcmin,
    T remapped to T(x + grad psi) by cubic-spline interpolation, then cut to the 512^2 grid, plus
    white noise of NOISE muK-arcmin (no beam);
  * the total spectrum C^tot of those maps (lensed CMB + noise), from the same NSIM skies whose
    reconstructions are analysed (each sky has its own random stream, so it can be drawn twice);
  * on the FFT grid, the normalisation A_L of the minimum-variance estimator, which is also N0; the
    weights use the lensed spectrum C^tot - noise, as analyses of real data do;
  * N0 by Monte Carlo: the estimator applied to Gaussian maps with spectrum C^tot and no lensing;
  * the raw spectrum <|psi_hat|^2> of NSIM lensed skies, and their cross-spectrum with the input;
  * N1 by Monte Carlo: pairs of CMB skies lensed by one shared psi, minus the same pairs lensed
    by two independent potentials;
  * N1 by direct evaluation of its four-dimensional integral at the bin centres.

Writes: figures/chT1/lens_n0n1.pdf, figures/chT1/lens_maps.pdf, results/chT1/11_lens_n0n1.tex
Needs:  data/chT1/t1c_unlensed.npz (written by 06_rotated_spectra.py)
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from common import setup, savefig, save_numbers, rng_for, theory_line, progress, SERIES, DIV, DATA, SEED_SALT

N = 512
PIX = np.deg2rad(2.0 / 60.0)          # pixel side [rad]
AREA = (N * PIX) ** 2
ARCMIN = np.deg2rad(1.0 / 60.0)
NOISE = 5.0                           # white noise [muK-arcmin]
LMIN, LMAX = 30, 2500                 # CMB multipoles used by the estimator
NH = 2 * N                            # fine grid on which the sky is lensed
PIXH = PIX / 2
NSIM, NGAU, NPAIR = 400, 400, 400
# all simulation products; delete it to rerun the simulations
CACHE = DATA / "chT1" / f"lens_n0n1{'_salt' + SEED_SALT if SEED_SALT else ''}.npz"
EDGES = np.array([20, 60, 100, 150, 200, 275, 350, 450, 575, 700, 850, 1000, 1200, 1400, 1700, 2000])


def to_k(x):
    """Continuum Fourier transform on the grid: X_l = int d^2x e^{-il.x} X(x)."""
    return np.fft.fft2(x) * PIX**2


def to_x(k):
    """Inverse: X(x) = int d^2l/(2pi)^2 e^{il.x} X_l."""
    return np.fft.ifft2(k) / PIX**2


def simulate():
    z = np.load(DATA / "chT1" / "t1c_unlensed.npz")
    L1d = z["ell"].astype(float)
    k = 2 * np.pi * np.fft.fftfreq(N, d=PIX)
    lx, ly = np.meshgrid(k, k, indexing="xy")
    ell = np.hypot(lx, ly)
    dl = k[1]

    def on_grid(cl):
        out = np.interp(ell, L1d, cl, left=0.0, right=0.0)
        out[0, 0] = 0.0
        return out
    cpp = on_grid(z["cpp"])
    nl = (NOISE * ARCMIN) ** 2                            # white noise power [muK^2 sr]
    use = (ell >= LMIN) & (ell <= LMAX)

    gen = [rng_for("chT1", "11_lens_n0n1")]     # the stream gauss() draws from

    def own_stream(s):
        """Point gauss() at lensed sky s's own stream; returns the main stream to restore."""
        main, gen[0] = gen[0], rng_for("chT1", "11_lens_n0n1", stream=1 + s)
        return main

    def gauss(c, n=N):
        """A Gaussian field with spectrum c, in Fourier space: <|X_l|^2> = AREA c."""
        return np.sqrt(c * AREA) * np.fft.fft2(gen[0].standard_normal((n, n))) / n

    # the fine grid: same patch, twice the pixels; its modes with |l_x|, |l_y| below the coarse
    # Nyquist frequency are the coarse grid's modes
    kh = 2 * np.pi * np.fft.fftfreq(NH, d=PIXH)
    lxh, lyh = np.meshgrid(kh, kh, indexing="xy")
    ellh = np.hypot(lxh, lyh)
    sub = np.r_[0:N // 2, NH - N // 2:NH]
    cuh = np.interp(ellh, L1d, z["TTu"], left=0.0, right=0.0); cuh[0, 0] = 0.0
    cpph = np.interp(ellh, L1d, z["cpp"], left=0.0, right=0.0); cpph[0, 0] = 0.0
    iy, ix = np.meshgrid(np.arange(NH), np.arange(NH), indexing="ij")

    def sky():
        """An unlensed T and a potential psi on the fine grid (Fourier space)."""
        return gauss(cuh, NH), gauss(cpph, NH)

    def lens(tkh, pkh):
        """T(x + grad psi) on the fine grid, by cubic-spline interpolation; coarse Fourier modes."""
        t = np.real(np.fft.ifft2(tkh)) / PIXH**2
        dx = np.real(np.fft.ifft2(1j * lxh * pkh)) / PIXH**2
        dy = np.real(np.fft.ifft2(1j * lyh * pkh)) / PIXH**2
        tl = map_coordinates(t, [iy + dy / PIXH, ix + dx / PIXH], order=3, mode="grid-wrap")
        return (np.fft.fft2(tl) * PIXH**2)[np.ix_(sub, sub)]

    def coarse(kh_):
        return kh_[np.ix_(sub, sub)]

    def noise():
        return gauss(np.full_like(ell, nl))

    # ---- bins in L ----
    which = np.digitize(ell, EDGES) - 1
    nb = EDGES.size - 1
    ok = (which >= 0) & (which < nb)
    cnt = np.bincount(which[ok], minlength=nb)
    Lcen = np.bincount(which[ok], ell[ok], nb) / cnt

    def binned(m):
        return np.bincount(which[ok], np.real(m[ok]), nb) / cnt

    # ---- 1. total spectrum of the observed maps, ring by ring ----
    ring = np.rint(ell / dl).astype(int)
    nring = ring.max() + 1
    rcnt = np.bincount(ring.ravel(), minlength=nring)
    pw = np.zeros(nring)
    for s in range(NSIM):
        main = own_stream(s)
        d = lens(*sky()) + noise()
        gen[0] = main
        pw += np.bincount(ring.ravel(), (np.abs(d) ** 2).ravel() / AREA, nring)
        progress("C^tot of the lensed maps", s + 1, NSIM)
    rl = np.bincount(ring.ravel(), ell.ravel(), nring) / np.maximum(rcnt, 1)
    good = rcnt > 0
    ctot_ring = pw[good] / rcnt[good] / NSIM
    rl = rl[good]
    ctot = np.interp(ell, rl, ctot_ring)
    cw = np.where(ell > 0, ctot - nl, 0.0)        # lensed CMB spectrum: the weights' C_l
    cu = on_grid(z["TTu"])
    lensed_extra = float(np.median((cw / np.where(cu > 0, cu, 1) - 1)[(ell > 1800) & (ell < 2200)]))

    # ---- 2. estimator and its normalisation on the grid ----
    def filt(c):
        return np.where(use, 1.0 / c, 0.0)

    ivt = filt(ctot)

    def qe_half(xk, yk):
        """int d^2l1/(2pi)^2 (L.l1) C_l1 Xbar_l1 Ybar_{L-l1}, with Xbar = X/C^tot on the used modes."""
        yb = to_x(ivt * yk)
        s = sum(a * to_k(to_x(1j * a * cw * ivt * xk) * yb) for a in (lx, ly))
        return -1j * s

    def conv(g, h):
        return to_k(to_x(g) * to_x(h))

    def norm(ivc):
        """A_L^{-1} = int f^2 / (2 C1 C2) = int [(L.l1)^2 C1^2 + (L.l1) C1 (L.l2) C2] / (C1 C2)."""
        a, b = cw * ivc, ivc
        tot = (lx * lx * conv(lx * lx * cw * a, b) + 2 * lx * ly * conv(lx * ly * cw * a, b)
               + ly * ly * conv(ly * ly * cw * a, b)
               + lx * lx * conv(lx * a, lx * a) + 2 * lx * ly * conv(lx * a, ly * a)
               + ly * ly * conv(ly * a, ly * a))
        tot = np.real(tot)
        A = np.where(tot > 0, 1.0 / np.where(tot > 0, tot, 1), 0.0)
        A[0, 0] = 0.0
        return A

    A = norm(ivt)
    A_cmb = norm(filt(ctot - nl))               # the same with the instrument noise removed

    def est(xk, yk):
        """Normalised estimate of psi_L from the pair of maps (X, Y)."""
        return A * 0.5 * (qe_half(xk, yk) + qe_half(yk, xk))

    def est_auto(xk):
        return A * qe_half(xk, xk)

    # ---- 3. lensed skies: raw spectrum, cross with the input, input power ----
    keys = ["raw", "cross", "inp", "rawminus"]
    S = {q: np.zeros(nb) for q in keys}
    S2 = {q: np.zeros(nb) for q in keys}
    A_b = binned(A)
    example = None
    for s in range(NSIM):
        main = own_stream(s)                   # the same sky as in step 1
        tkh, pkh = sky()
        pk = coarse(pkh)
        d = lens(tkh, pkh) + noise()
        gen[0] = main
        ph = est_auto(d)
        vals = {"raw": binned(np.abs(ph) ** 2 / AREA), "cross": binned(ph * np.conj(pk) / AREA),
                "inp": binned(np.abs(pk) ** 2 / AREA)}
        vals["rawminus"] = vals["raw"] - vals["inp"] - A_b
        for q in keys:
            S[q] += vals[q]; S2[q] += vals[q] ** 2
        if s == 0:
            example = (pk, ph)
        progress("lensed skies: raw spectrum", s + 1, NSIM)

    # ---- 4. Gaussian skies with the same total spectrum: N0 by Monte Carlo ----
    G = np.zeros(nb); G2 = np.zeros(nb)
    for s in range(NGAU):
        ph = est_auto(gauss(ctot))
        v = binned(np.abs(ph) ** 2 / AREA)
        G += v; G2 += v**2
        progress("Gaussian skies: N0", s + 1, NGAU)

    # ---- 5. N1 by Monte Carlo: shared psi minus independent psi ----
    def pair_power(a, b):
        """C[a,b,a,b] + C[a,b,b,a]; the weights are symmetric, so est(a, b) = est(b, a) and the two are equal."""
        return binned(2 * np.abs(est(a, b)) ** 2 / AREA)

    P = np.zeros(nb); P2 = np.zeros(nb)
    for s in range(NPAIR):
        (ta, p1), (tb, p2) = sky(), sky()
        na, nbk = noise(), noise()
        da = lens(ta, p1) + na
        shared = pair_power(da, lens(tb, p1) + nbk)
        indep = pair_power(da, lens(tb, p2) + nbk)
        v = shared - indep
        P += v; P2 += v**2
        progress("sky pairs: N1", s + 1, NPAIR)

    def mean_sem(s1, s2, n):
        m = s1 / n
        return m, np.sqrt(np.maximum(s2 / n - m**2, 0) / (n - 1))
    mean = {q: mean_sem(S[q], S2[q], NSIM) for q in keys}
    n0_mc, n0_mc_e = mean_sem(G, G2, NGAU)
    n1_mc, n1_mc_e = mean_sem(P, P2, NPAIR)
    cpp_b = binned(cpp)

    # ---- 6. N1 and A_L from their integrals (polar quadrature, continuous l) ----
    cu_f = lambda x: np.interp(x, rl, ctot_ring - nl)      # lensed spectrum, as in the weights
    ct_f = lambda x: np.interp(x, rl, ctot_ring)
    cp_f = lambda x: np.interp(x, L1d, z["cpp"], left=0.0, right=0.0)
    r = np.arange(LMIN + 12.5, LMAX, 25.0)
    th = (np.arange(128) + 0.5) * 2 * np.pi / 128
    R, TH = np.meshgrid(r, th, indexing="ij")
    px, py = (R * np.cos(TH)).ravel(), (R * np.sin(TH)).ravel()
    pw1 = (R * 25.0 * 2 * np.pi / 128).ravel() / (2 * np.pi) ** 2
    rd = np.geomspace(2.0, 5000.0, 90)                     # Delta = l1 + l1', fine near zero
    thd = (np.arange(64) + 0.5) * 2 * np.pi / 64
    RD, THD = np.meshgrid(rd, thd, indexing="ij")
    dx, dy = (RD * np.cos(THD)).ravel(), (RD * np.sin(THD)).ravel()
    dw = (RD * np.gradient(rd)[:, None] * 2 * np.pi / 64).ravel() / (2 * np.pi) ** 2
    cpd = cp_f(np.hypot(dx, dy))

    def f(ax_, ay_, bx_, by_):
        sx, sy = ax_ + bx_, ay_ + by_
        return (sx * ax_ + sy * ay_) * cu_f(np.hypot(ax_, ay_)) + (sx * bx_ + sy * by_) * cu_f(np.hypot(bx_, by_))

    def F(ax_, ay_, bx_, by_):
        ma, mb = np.hypot(ax_, ay_), np.hypot(bx_, by_)
        w = (ma >= LMIN) & (ma <= LMAX) & (mb >= LMIN) & (mb <= LMAX)
        return np.where(w, f(ax_, ay_, bx_, by_) / (2 * ct_f(ma) * ct_f(mb)), 0.0)

    n1_th = np.zeros(nb); A_th = np.zeros(nb)
    # which L bin each Delta = l1 - l1' falls in (nb = outside the bins)
    dbin = np.digitize(np.hypot(dx, dy), EDGES) - 1
    dbin = np.where((dbin >= 0) & (dbin < nb), dbin, nb)
    contrib = np.zeros((nb, nb + 1))     # N1 of bin b drawn from C^psipsi in bin b' (last column: outside)
    for b, Lb in enumerate(Lcen):
        # l1 = p, l2 = L - p
        F1 = F(px, py, Lb - px, -py)
        A_th[b] = 1.0 / np.sum(pw1 * F1 * f(px, py, Lb - px, -py))
        tot = np.zeros(nb + 1)
        for i in np.nonzero(F1)[0]:
            # l1' = Delta - l1, l2' = -L - l1'
            q1x, q1y = dx - px[i], dy - py[i]
            q2x, q2y = -Lb - q1x, -q1y
            Fp = F(q1x, q1y, q2x, q2y)
            g = f(px[i], py[i], q1x, q1y) * f(Lb - px[i], -py[i], q2x, q2y)
            tot += pw1[i] * F1[i] * np.bincount(dbin, dw * Fp * g * cpd, nb + 1)
        contrib[b] = 2 * A_th[b] ** 2 * tot
        n1_th[b] = contrib[b].sum()
        progress("N1 integral, L bins", b + 1, nb)
    # A_L is steep, so its bin average exceeds its value at the mean L; N1/A_L varies slowly
    n1_b = n1_th / A_th * A_b
    # N1_b = sum_b' K_bb' C_b' + N1_out_b, exact for the fiducial shape of C^psipsi inside each bin
    contrib *= (A_b / A_th)[:, None]
    K = contrib[:, :nb] / cpp_b[None, :]
    n1_out = contrib[:, nb]
    assert np.allclose(K @ cpp_b + n1_out, n1_b)
    # the grid's A_L against the integral at single modes on the l_x axis
    kt = np.array([3, 10, 30, 60, 90])
    A_axis = A[0, kt]
    A_axis_th = np.array([1.0 / np.sum(pw1 * F(px, py, Lt - px, -py) * f(px, py, Lt - px, -py)) for Lt in kt * dl])

    return dict(Lcen=Lcen, A_b=A_b, cpp_b=cpp_b, n1_b=n1_b, raw=mean["raw"][0], raw_e=mean["raw"][1],
                rm=mean["rawminus"][0], rm_e=mean["rawminus"][1], cross=mean["cross"][0], cross_e=mean["cross"][1],
                inp=mean["inp"][0], n0_mc=n0_mc, n0_mc_e=n0_mc_e, n1_mc=n1_mc, n1_mc_e=n1_mc_e,
                A_cmb_b=binned(A_cmb), A_axis=A_axis, A_axis_th=A_axis_th, A_th=A_th,
                K=K, n1_out=n1_out, lensed_extra=lensed_extra, pk=example[0], ph=example[1], A=A, cpp=cpp, ell=ell)


def main():
    if CACHE.exists():
        r = dict(np.load(CACHE))
    else:
        r = simulate()
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(CACHE, **r)
    Lcen, A_b, cpp_b, n1_b, ell = r["Lcen"], r["A_b"], r["cpp_b"], r["n1_b"], r["ell"]
    n0_mc, n0_mc_e, n1_mc, n1_mc_e = r["n0_mc"], r["n0_mc_e"], r["n1_mc"], r["n1_mc_e"]
    A_axis, A_axis_th, A_th, lensed_extra = r["A_axis"], r["A_axis_th"], r["A_th"], float(r["lensed_extra"])
    nb = Lcen.size

    # ---- figures ----
    fac = (Lcen * (Lcen + 1)) ** 2 / (2 * np.pi)
    setup(7.2, 3.2)
    fig, axs = plt.subplots(1, 2)
    ax = axs[0]
    raw, raw_e = r["raw"], r["raw_e"]
    ax.errorbar(Lcen, fac * raw, fac * raw_e, fmt="o", ms=3, color=SERIES[0], label=r"$\langle\hat C^{\hat\psi\hat\psi}_L\rangle$, lensed skies")
    ax.errorbar(Lcen * 1.02, fac * n0_mc, fac * n0_mc_e, fmt="s", ms=3, color=SERIES[1], label="Gaussian skies (no lensing)")
    ax.plot(Lcen, fac * A_b, color=SERIES[1], lw=1.2, label=r"$N^{(0)}_L=A_L$")
    ax.plot(Lcen, fac * (cpp_b + A_b + n1_b), color=SERIES[0], lw=1.2, label=r"$C^{\psi\psi}_L+N^{(0)}_L+N^{(1)}_L$")
    theory_line(ax, Lcen, fac * cpp_b, label=r"$C^{\psi\psi}_L$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$L$"); ax.set_title(r"$[L(L+1)]^2C_L/2\pi$", fontsize=9)
    ax.legend(fontsize=6.5, loc="lower left")
    ax = axs[1]
    rm, rm_e = r["rm"], r["rm_e"]
    ax.errorbar(Lcen, 100 * n1_mc / cpp_b, 100 * n1_mc_e / cpp_b, fmt="o", ms=3, color=SERIES[2], label="shared-$\\psi$ sky pairs")
    ax.errorbar(Lcen * 1.02, 100 * rm / cpp_b, 100 * rm_e / cpp_b, fmt="s", ms=3, color=SERIES[0], mfc="none",
                label=r"raw $-\,C^{\psi\psi}_L-N^{(0)}_L$")
    theory_line(ax, Lcen, 100 * n1_b / cpp_b, label="integral")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xscale("log"); ax.set_xlabel(r"$L$"); ax.set_title(r"$N^{(1)}_L/C^{\psi\psi}_L$ [%]", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    savefig(fig, "chT1", "lens_n0n1")

    # maps: input and Wiener-filtered reconstruction, as convergence kappa_L = -L^2 psi_L / 2,
    # both also smoothed with a Gaussian of 1 degree FWHM
    pk, ph, A, cpp = r["pk"], r["ph"], r["A"], r["cpp"]
    sig = np.deg2rad(1.0) / np.sqrt(8 * np.log(2))
    wf = np.where(ell > 0, cpp / np.where(cpp + A > 0, cpp + A, 1), 0.0) * np.exp(-0.5 * (ell * sig) ** 2)
    kin = np.real(to_x(-0.5 * ell**2 * pk * wf))
    krec = np.real(to_x(-0.5 * ell**2 * ph * wf))
    vmax = 3 * np.std(kin)
    setup(7.0, 3.3)
    fig, axs = plt.subplots(1, 2)
    ext = [0, N * 2 / 60, 0, N * 2 / 60]
    for ax, m, t in zip(axs, (kin, krec), ("input $\\kappa$", "reconstructed $\\hat\\kappa$")):
        im = ax.imshow(m, cmap=DIV, vmin=-vmax, vmax=vmax, extent=ext, origin="lower")
        ax.set_title(t, fontsize=9); ax.set_xlabel("degrees"); ax.grid(False)
    axs[0].set_ylabel("degrees")
    fig.colorbar(im, ax=axs, shrink=0.8, pad=0.02)
    savefig(fig, "chT1", "lens_maps")

    # ---- the N1 equation: d = (I + K) c + N1_out, with d the raw spectrum minus N0 ----
    K, n1_out = r["K"], r["n1_out"]
    d = r["raw"] - A_b
    c_star = np.linalg.solve(np.eye(nb) + K, d - n1_out)
    rho = float(np.max(np.abs(np.linalg.eigvals(K))))
    starts = {"fid": cpp_b, "half": 0.5 * cpp_b, "double": 2.0 * cpp_b}
    errs = {}
    for name, c in starts.items():
        e = []
        for _ in range(12):
            e.append(float(np.max(np.abs(c - c_star) / c_star)))
            c = d - n1_out - K @ c
        errs[name] = np.array(e)
    rate = float(np.median(errs["double"][4:9] / errs["double"][3:8]))
    iters = int(np.argmax(errs["double"] < 1e-3))
    one_step = float(np.max(np.abs(K @ (0.1 * cpp_b)) / cpp_b))       # error left by a 10% wrong fiducial
    sol_chi = float(np.sum(((c_star - r["inp"]) / r["raw_e"]) ** 2 * (Lcen < 1000)))
    setup(7.2, 3.0)
    fig, axs = plt.subplots(1, 2)
    im = axs[0].imshow(K * cpp_b[None, :] / cpp_b[:, None], origin="lower", cmap="Blues",
                       extent=[-0.5, nb - 0.5, -0.5, nb - 0.5])
    axs[0].set_xlabel(r"bin of $L'$"); axs[0].set_ylabel(r"bin of $L$"); axs[0].grid(False)
    axs[0].set_title(r"$K_{LL'}C^{\psi\psi}_{L'}/C^{\psi\psi}_L$", fontsize=9)
    fig.colorbar(im, ax=axs[0], shrink=0.8)
    for (name, e), col, lab in zip(errs.items(), SERIES, ["start: fiducial", "start: fiducial/2", "start: 2 x fiducial"]):
        axs[1].semilogy(np.arange(e.size), np.maximum(e, 1e-16), "o-", ms=3, color=col, label=lab)
    axs[1].semilogy(np.arange(12), errs["double"][0] * rho ** np.arange(12), "k--", lw=1, label=r"$\rho(K)^n$")
    axs[1].set_xlabel("iteration $n$"); axs[1].set_title("largest relative error", fontsize=9)
    axs[1].legend(fontsize=6.5)
    for ax in (axs[0].xaxis, axs[0].yaxis, axs[1].xaxis):
        ax.set_major_locator(plt.MaxNLocator(integer=True))
    fig.tight_layout()
    savefig(fig, "chT1", "lens_n1_iter")

    # ---- numbers ----
    cross, cross_e, inp = r["cross"], r["cross_e"], r["inp"]
    w = (Lcen > 40)
    resp = float(np.sum((cross / inp)[w] / ((cross_e / inp)[w]) ** 2) / np.sum(1 / ((cross_e / inp)[w]) ** 2))
    resp_e = float(1 / np.sqrt(np.sum(1 / ((cross_e / inp)[w]) ** 2)))
    n0_dev = float(np.max(np.abs(n0_mc / A_b - 1)))
    n0_chi = float(np.sum(((n0_mc - A_b) / n0_mc_e) ** 2))
    n1_chi = float(np.sum(((n1_mc - n1_b) / n1_mc_e) ** 2))
    n1_sig = float(np.sqrt(np.sum((n1_mc / n1_mc_e) ** 2)))
    rm_chi = float(np.sum(((rm - n1_b) / rm_e) ** 2))
    low = Lcen < 1000                      # where N0 is at most about twenty times the signal
    rm_chi_low = float(np.sum((((rm - n1_b) / rm_e) ** 2)[low]))
    frac = n1_b / cpp_b
    ib = int(np.argmax(frac))
    cross_L = float(np.interp(0, np.log(cpp_b / A_b)[::-1], Lcen[::-1]))   # where C^psipsi = N0
    noise_part = 1 - r["A_cmb_b"] / A_b
    # bias of a fitted amplitude A C^psipsi if N1 were left in: Fisher weights per bin
    cnt = np.array([np.sum((ell >= lo) & (ell < hi)) for lo, hi in zip(EDGES[:-1], EDGES[1:])])
    wfis = cnt * cpp_b**2 / (cpp_b + A_b + n1_b) ** 2
    amp_bias = float(np.sum(wfis * n1_b / cpp_b) / np.sum(wfis))
    i100, i1000 = int(np.argmin(np.abs(Lcen - 100))), int(np.argmin(np.abs(Lcen - 1000)))
    print("Lcen", np.round(Lcen))
    print("A grid / A integral on axis", np.round(A_axis / A_axis_th, 4), "bins", np.round(A_b / A_th, 4))
    print("N0 MC / A", np.round(n0_mc / A_b, 4))
    print("N1 MC / C", np.round(n1_mc / cpp_b, 4), "+-", np.round(n1_mc_e / cpp_b, 4))
    print("N1 th / C", np.round(n1_b / cpp_b, 4))
    print("raw-C-N0 / C", np.round(rm / cpp_b, 4), "+-", np.round(rm_e / cpp_b, 4))
    print("N0 / C", np.round(A_b / cpp_b, 3))
    print("noise part of N0", np.round(noise_part, 3))
    print("resp", resp, resp_e, "chi2 N0", n0_chi, "N1", n1_chi, "rm", rm_chi, "N1 sig", n1_sig, "dof", nb)
    print("lensed excess of C^tot near l=2000", lensed_extra, "amplitude bias from N1", amp_bias)
    print("rho(K)", rho, "rate", rate, "iters to 1e-3", iters, "one-step 10% err", one_step, "sol chi2 L<1000", sol_chi)
    print("K row sums (N1/C)", np.round(K @ cpp_b / cpp_b, 3), "diag", np.round(np.diag(K), 3))
    for k_, v_ in errs.items(): print(k_, np.array2string(v_, precision=2))
    save_numbers("chT1", "11_lens_n0n1", {
        "TOneRN": N, "TOneRNoise": NOISE, "TOneRLmin": LMIN, "TOneRLmax": LMAX,
        "TOneRNsim": NSIM, "TOneRNgau": NGAU, "TOneRNpair": NPAIR, "TOneRNbin": nb,
        "TOneRResp": f"{resp:.3f}", "TOneRRespErr": f"{resp_e:.3f}",
        "TOneRNzeroDev": f"{100 * n0_dev:.1f}", "TOneRNzeroChi": f"{n0_chi:.1f}",
        "TOneRNoneChi": f"{n1_chi:.1f}", "TOneRRawChi": f"{rm_chi:.1f}",
        "TOneRRawChiLow": f"{rm_chi_low:.1f}", "TOneRNbinLow": int(low.sum()), "TOneRNoneSig": f"{n1_sig:.0f}",
        "TOneRNoneMax": f"{100 * frac[ib]:.1f}", "TOneRNoneMaxL": f"{Lcen[ib]:.0f}",
        "TOneRNoneHundred": f"{100 * frac[i100]:.1f}", "TOneRNoneThousand": f"{100 * frac[i1000]:.1f}",
        "TOneRNzeroOverCHundred": f"{A_b[i100] / cpp_b[i100]:.2f}",
        "TOneRNzeroOverCThousand": f"{A_b[i1000] / cpp_b[i1000]:.1f}",
        "TOneRCrossL": f"{cross_L:.0f}",
        "TOneRNoisePartHundred": f"{100 * noise_part[i100]:.0f}",
        "TOneRNoisePartThousand": f"{100 * noise_part[i1000]:.0f}",
        "TOneRLensedExtra": f"{100 * lensed_extra:.0f}",
        "TOneRAmpBias": f"{100 * amp_bias:.1f}",
        "TOneRRho": f"{rho:.2f}", "TOneRIterRate": f"{rate:.2f}", "TOneRIterN": iters,
        "TOneROneStep": f"{100 * one_step:.1f}",
        "TOneROneStepAmp": f"{10 * amp_bias:.1f}", "TOneRSolChi": f"{sol_chi:.1f}",
        "TOneRAgridDev": f"{100 * float(np.max(np.abs(A_axis / A_axis_th - 1))):.1f}",
    })


if __name__ == "__main__":
    main()
