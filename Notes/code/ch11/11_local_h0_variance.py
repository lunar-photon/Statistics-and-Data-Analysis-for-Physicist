"""11_local_h0_variance.py -- cosmic variance of a locally measured Hubble constant.

Question: an observer fits H from radial peculiar velocities of tracers inside a sphere of
radius R. Because the tracers sit in a fluctuating density field, the answer differs from
the global H0. By how much, as a function of R, and for which estimator?

Linear theory: delta H / H0 = -(f/3) delta_W, with delta_W the density contrast averaged
with the estimator's window. Windows in Fourier space (x = kR):
    top hat (mass inside R)                  W_TH = 3 j1(x)/x
    least-squares slope sum(u_r r)/sum(r^2)  W_LS = 15 j2(x)/x^2
    mean of ratios <u_r / r>                 W_MR = 9 (Si(x) - sin x)/x^3
    sigma^2(dH/H) = (f/3)^2 int dk k^2 P(k) W(kR)^2 / (2 pi^2).
Reproduces: Wojtak et al. (2014) Table 1 "linear model" (their eq. 3, WMAP-like cosmology
via CAMB) and Marra et al. (2013) Table I, cases I and III (top hat averaged over a
supernova redshift distribution). Verifies the least-squares window on Gaussian boxes.
Writes figures/ch11/local_h0_windows.pdf, figures/ch11/local_h0_sigma.pdf,
results/ch11/11_local_h0_variance.tex; caches data/ch11/pk_wojtak.npz.
Laptop scale: 4 boxes of 1500 Mpc/h, 256^3 cells, 250 observers each (~2 minutes).
"""
import pathlib
import sys

import numpy as np
from scipy.optimize import brentq, fsolve
from scipy.special import sici, spherical_jn
from scipy.stats import norm

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import DATA, SERIES, rng_for, save_numbers, savefig, setup, theory_line  # noqa: E402
from lib_lss import E_of_z, OMM, growth, linear_pk_z0  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

CH, NAME = "ch11", "11_local_h0_variance"
C_KMS = 299792.458


def W_TH(x):
    return 3 * spherical_jn(1, x) / x


def W_LS(x):
    return 15 * spherical_jn(2, x) / x ** 2


def W_MR(x):
    return 9 * (sici(x)[0] - np.sin(x)) / x ** 3


def sigma_H(R, k, P, f, W):
    """rms of delta H / H0 for window W at radius R [Mpc/h]."""
    lk = np.log(k)
    R = np.atleast_1d(R)
    return np.array([(f / 3) * np.sqrt(np.trapezoid(k ** 3 * P * W(k * r) ** 2, lk) / (2 * np.pi ** 2))
                     for r in R])


def pk_wojtak():
    """Linear P(k) at z=0 for the Jubilee/WMAP5 cosmology of Wojtak et al. (2014), cached."""
    path = DATA / CH / "pk_wojtak.npz"
    if path.exists():
        z = np.load(path)
        return z["k"], z["pk"]
    import camb
    h, om, ob, ns, s8 = 0.70, 0.27, 0.044, 0.96, 0.80
    pars = camb.set_params(H0=100 * h, ombh2=ob * h ** 2, omch2=(om - ob) * h ** 2, ns=ns,
                           As=2.1e-9, mnu=0.0, WantTransfer=True)
    pars.set_matter_power(redshifts=[0.0], kmax=50.0)
    res = camb.get_results(pars)
    kh, _, pk = res.get_matter_power_spectrum(minkh=1e-4, maxkh=49.0, npoints=2000)
    pk = pk[0] * (s8 / res.get_sigma8()[0]) ** 2
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, k=kh, pk=pk)
    return kh, pk


def comoving(z, om=OMM):
    """Comoving distance in Mpc/h for flat LCDM."""
    zz = np.linspace(0, z, 400)
    return (C_KMS / 100.0) * np.trapezoid(1 / E_of_z(zz, om), zz)


def wsn_fit(medians=((0.010, 0.025), (0.023, 0.033)), zmax=0.1):
    """Lognormal-in-z redshift distribution on [zmin, zmax] whose truncated medians match Marra's."""
    def trunc_median(mu, s, zmin):
        a, b = norm.cdf((np.log(zmin) - mu) / s), norm.cdf((np.log(zmax) - mu) / s)
        return np.exp(mu + s * norm.ppf(a + 0.5 * (b - a)))
    def eqs(p):
        mu, ls = p
        return [np.log(trunc_median(mu, np.exp(ls), zm)) - np.log(med) for zm, med in medians]
    mu, ls = fsolve(eqs, [np.log(0.025), np.log(0.6)])
    return mu, np.exp(ls)


def marra_sigma(k, P, f, zmin, zmax, mu, s, W=W_TH):
    zz = np.linspace(zmin, zmax, 200)
    w = np.exp(-0.5 * ((np.log(zz) - mu) / s) ** 2) / zz
    w /= np.trapezoid(w, zz)
    R = np.array([comoving(z) for z in zz])
    sig = sigma_H(R, k, P, f, W)
    return np.sqrt(np.trapezoid(w * sig ** 2, zz)), zz, w


def box_check(k, P, f, Rs, L=1500.0, N=256, nbox=4, nobs=250, npts=3000):
    rng = rng_for(CH, NAME)
    D = L / N
    k1 = 2 * np.pi * np.fft.fftfreq(N, d=D)
    kz = 2 * np.pi * np.fft.rfftfreq(N, d=D)
    KX, KY, KZ = np.meshgrid(k1, k1, kz, indexing="ij")
    K2 = KX ** 2 + KY ** 2 + KZ ** 2
    Kmag = np.sqrt(K2)
    Pg = np.where(Kmag > 0, np.interp(np.where(Kmag > 0, Kmag, 1.0), k, P), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_k2 = np.where(K2 > 0, 1.0 / K2, 0.0)
    wts = np.full(Kmag.shape, 2.0)
    wts[..., 0] = 1.0
    wts[..., -1] = 1.0
    V = L ** 3
    # exact grid expectation of sigma^2(dH/H) for the least-squares window
    grid_th = []
    for R in Rs:
        x = np.where(Kmag > 0, Kmag * R, 1.0)
        Wg = np.where(Kmag > 0, W_LS(x), 0.0)
        grid_th.append((f / 3) * np.sqrt(np.sum(wts * Pg * Wg ** 2) / V))
    amp = np.sqrt(Pg / D ** 3)
    meas = {R: [] for R in Rs}
    for b in range(nbox):
        dk = np.fft.rfftn(rng.standard_normal((N, N, N))) * amp
        u = [np.fft.irfftn(1j * 100.0 * f * Kc * inv_k2 * dk, s=(N, N, N), axes=(0, 1, 2))
             for Kc in (KX, KY, KZ)]
        del dk
        for _ in range(nobs):
            x0 = rng.uniform(0, L, 3)
            for R in Rs:
                rr = R * rng.random(npts) ** (1 / 3)
                mu_ = rng.uniform(-1, 1, npts)
                ph = rng.uniform(0, 2 * np.pi, npts)
                sn = np.sqrt(1 - mu_ ** 2)
                nhat = np.column_stack([sn * np.cos(ph), sn * np.sin(ph), mu_])
                pos = (x0 + rr[:, None] * nhat) % L
                idx = np.floor(pos / D).astype(int) % N
                ur = sum(u[c][idx[:, 0], idx[:, 1], idx[:, 2]] * nhat[:, c] for c in range(3))
                meas[R].append(np.sum(ur * rr) / np.sum(rr ** 2) / 100.0)
        del u
    return np.array(grid_th), {R: np.array(v) for R, v in meas.items()}


def main():
    setup()
    k, P, s8, _ = linear_pk_z0()
    f = float(growth(0.0)[1][0])
    Rgrid = np.geomspace(20, 400, 60)
    sTH, sLS, sMR = (sigma_H(Rgrid, k, P, f, W) for W in (W_TH, W_LS, W_MR))

    # ---- Wojtak et al. 2014 (WMAP-like cosmology, f = Om^0.55)
    kw, Pw = pk_wojtak()
    fw = 0.27 ** 0.55
    Rw = np.array([50.0, 75.0, 150.0])
    w_MR = 100 * sigma_H(Rw, kw, Pw, fw, W_MR)
    w_LS = 100 * sigma_H(Rw, kw, Pw, fw, W_LS)

    # ---- Marra et al. 2013 (top hat averaged over a supernova distribution)
    mu, s = wsn_fit()
    m1, zz1, w1 = marra_sigma(k, P, f, 0.010, 0.1, mu, s)
    m3, zz3, w3 = marra_sigma(k, P, f, 0.023, 0.1, mu, s)
    m1LS, _, _ = marra_sigma(k, P, f, 0.010, 0.1, mu, s, W_LS)
    m3LS, _, _ = marra_sigma(k, P, f, 0.023, 0.1, mu, s, W_LS)
    # a SH0ES-like range
    mS, _, _ = marra_sigma(k, P, f, 0.023, 0.15, mu, s)
    mSLS, _, _ = marra_sigma(k, P, f, 0.023, 0.15, mu, s, W_LS)

    # ---- tension: what contrast would turn 67.36 into 73.04?
    gap = 73.04 / 67.36 - 1
    deltaW = -3 * gap / f
    nsig = gap / mSLS

    # ---- box check of the least-squares window
    Rs = [50.0, 100.0, 150.0, 200.0, 300.0]
    grid_th, meas = box_check(k, P, f, Rs)
    box_sd = np.array([meas[R].std() for R in Rs])
    box_mean = np.array([meas[R].mean() for R in Rs])
    cont_LS = sigma_H(np.array(Rs), k, P, f, W_LS)

    # ---------------- figure 1: windows in real and Fourier space
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
    y = np.linspace(0, 1, 300)                   # r/R
    ax[0].plot(y, np.ones_like(y), color=SERIES[0], label="top hat (mass inside $R$)")
    ax[0].plot(y, 2.5 * (1 - y ** 2), color=SERIES[1], label=r"least squares: $\propto R^2-r^2$")
    ax[0].plot(y[1:], 3 * np.log(1 / y[1:]), color=SERIES[2], label=r"mean of ratios: $\propto\ln(R/r)$")
    ax[0].set_ylim(0, 6)
    ax[0].set_xlabel(r"$r/R$")
    ax[0].set_ylabel("weight per unit volume (mean 1)")
    ax[0].legend(fontsize=8)
    x = np.geomspace(0.05, 30, 400)
    for W, c, lab in ((W_TH, SERIES[0], r"$3j_1(x)/x$"), (W_LS, SERIES[1], r"$15j_2(x)/x^2$"),
                      (W_MR, SERIES[2], r"$9[\mathrm{Si}(x)-\sin x]/x^3$")):
        ax[1].semilogx(x, W(x), color=c, label=lab)
    ax[1].axhline(0, color="0.6", lw=0.6)
    ax[1].set_xlabel(r"$x=kR$")
    ax[1].set_ylabel(r"$\tilde W(kR)$")
    ax[1].set_ylim(-0.12, 1.45)
    ax[1].legend(fontsize=8, loc="upper right")
    savefig(fig, CH, "local_h0_windows")

    # ---------------- figure 2: sigma(R), box check, literature
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ax[0].loglog(Rgrid, 100 * sTH, color=SERIES[0], label="top hat")
    ax[0].loglog(Rgrid, 100 * sLS, color=SERIES[1], label="least-squares slope")
    ax[0].loglog(Rgrid, 100 * sMR, color=SERIES[2], label="mean of ratios")
    ax[0].errorbar(Rs, 100 * box_sd, yerr=100 * box_sd / np.sqrt(2 * len(meas[Rs[0]])), fmt="o",
                   color=SERIES[1], mfc="white", ms=5, label="Gaussian boxes, least squares")
    ax[0].plot(Rs, 100 * grid_th, "x", color="k", ms=6, label="grid-mode sum")
    ax[0].set_xlabel(r"radius $R$ [Mpc/$h$]")
    ax[0].set_ylabel(r"rms of $\delta H/H_0$ [%]")
    ax[0].legend(fontsize=7.5)
    ax[0].set_ylim(0.1, 15)
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    ax[0].set_xticks([20, 50, 100, 200, 400])
    ax[0].xaxis.set_major_formatter(ScalarFormatter())
    ax[0].xaxis.set_minor_formatter(NullFormatter())
    # literature comparison bars
    labels = ["W14\n50", "W14\n75", "W14\n150", "M13\nI", "M13\nIII"]
    paper = [3.5, 2.2, 0.9, 2.1, 1.2]
    ours = list(w_MR) + [100 * m1, 100 * m3]
    xs = np.arange(len(labels))
    ax[1].bar(xs - 0.2, paper, width=0.4, color=SERIES[3], label="paper")
    ax[1].bar(xs + 0.2, ours, width=0.4, color=SERIES[0], label="ours (their window)")
    ax[1].plot(xs[:3] + 0.2, w_LS, "v", color=SERIES[1], label="ours, least-squares window")
    ax[1].plot(xs[3:] + 0.2, [100 * m1LS, 100 * m3LS], "v", color=SERIES[1])
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].set_ylabel(r"rms of $\delta H/H_0$ [%]")
    ax[1].set_ylim(0, ax[1].get_ylim()[1] * 1.3)
    ax[1].legend(fontsize=8, loc="upper right")
    savefig(fig, CH, "local_h0_sigma")

    iR = {R: int(np.argmin(np.abs(Rgrid - R))) for R in (50, 100, 150, 300)}
    out = {"FBLhF": f"{f:.3f}",
           "FBLhTHFifty": f"{100*sigma_H(50.,k,P,f,W_TH)[0]:.2f}",
           "FBLhLSFifty": f"{100*sigma_H(50.,k,P,f,W_LS)[0]:.2f}",
           "FBLhMRFifty": f"{100*sigma_H(50.,k,P,f,W_MR)[0]:.2f}",
           "FBLhTHHund": f"{100*sigma_H(100.,k,P,f,W_TH)[0]:.2f}",
           "FBLhLSHund": f"{100*sigma_H(100.,k,P,f,W_LS)[0]:.2f}",
           "FBLhMRHund": f"{100*sigma_H(100.,k,P,f,W_MR)[0]:.2f}",
           "FBLhLSOneFifty": f"{100*sigma_H(150.,k,P,f,W_LS)[0]:.2f}",
           "FBLhLSThreeHund": f"{100*sigma_H(300.,k,P,f,W_LS)[0]:.2f}",
           "FBLhWMRa": f"{w_MR[0]:.2f}", "FBLhWMRb": f"{w_MR[1]:.2f}", "FBLhWMRc": f"{w_MR[2]:.2f}",
           "FBLhWLSa": f"{w_LS[0]:.2f}", "FBLhWLSb": f"{w_LS[1]:.2f}", "FBLhWLSc": f"{w_LS[2]:.2f}",
           "FBLhWf": f"{fw:.3f}",
           "FBLhMuz": f"{np.exp(mu):.4f}", "FBLhSz": f"{s:.2f}",
           "FBLhMone": f"{100*m1:.2f}", "FBLhMthree": f"{100*m3:.2f}",
           "FBLhMoneLS": f"{100*m1LS:.2f}", "FBLhMthreeLS": f"{100*m3LS:.2f}",
           "FBLhShoes": f"{100*mS:.2f}", "FBLhShoesLS": f"{100*mSLS:.2f}",
           "FBLhGap": f"{100*gap:.1f}", "FBLhDeltaW": f"{deltaW:.2f}", "FBLhNsig": f"{nsig:.1f}",
           "FBLhBoxL": "1500", "FBLhBoxN": 256, "FBLhNobs": len(meas[Rs[0]]), "FBLhNpts": 3000}
    for R, nm in zip(Rs, ["Fifty", "Hund", "OneFifty", "TwoHund", "ThreeHund"]):
        i = Rs.index(R)
        out[f"FBLhBox{nm}"] = f"{100*box_sd[i]:.2f}"
        out[f"FBLhGrid{nm}"] = f"{100*grid_th[i]:.2f}"
        out[f"FBLhCont{nm}"] = f"{100*cont_LS[i]:.2f}"
        out[f"FBLhBoxMean{nm}"] = f"{100*box_mean[i]:.2f}"
    save_numbers(CH, NAME, out)
    print("ours (fiducial) TH/LS/MR at 50,100:", out["FBLhTHFifty"], out["FBLhLSFifty"], out["FBLhMRFifty"],
          out["FBLhTHHund"], out["FBLhLSHund"], out["FBLhMRHund"])
    print("Wojtak MR:", w_MR, "LS:", w_LS, " paper 3.5 2.2 0.9")
    print(f"Marra: W_SN median {np.exp(mu):.4f} s {s:.2f}; I {100*m1:.2f} (2.1) III {100*m3:.2f} (1.2); "
          f"LS {100*m1LS:.2f} {100*m3LS:.2f}; SH0ES-like TH {100*mS:.2f} LS {100*mSLS:.2f}")
    print(f"gap {100*gap:.1f}% deltaW {deltaW:.2f} nsig {nsig:.1f}")
    print("box sd", 100 * box_sd, "grid", 100 * grid_th, "cont", 100 * cont_LS, "mean", 100 * box_mean)


if __name__ == "__main__":
    main()
