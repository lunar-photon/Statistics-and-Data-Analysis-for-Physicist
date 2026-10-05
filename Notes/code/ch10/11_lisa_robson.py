"""11_lisa_robson.py -- reproduce the LISA sensitivity curve and the worked numbers of Robson, Cornish & Liu (2019).

Question: does our implementation of their eqs. (10)-(14) give their Figure 4 (sqrt(S_n) with the 4-year Galactic
confusion noise, and sqrt(S_c) alone), their transfer frequency f* = 19.09 mHz, the strain amplitude
h_GB = 2.8e-18 Hz^{-1/2} and SNR = 140 of the verification binary SDSS J0651+2844 (D_L ~ 1 kpc, 0.5 + 0.25 Msun,
f = 2.6 mHz, 4 years, their eqs. 26-27), the 16 -> 29 mHz sweep of a GW150914-like binary 5 years before merger,
and the 2.93e-5 Hz frequency one year before merger of an equal-mass 1e6 Msun binary at z = 3 (detector-frame mass)?
Their Figure 4 is digitised from the PDF (pixel colours inside the calibrated plot frame) and overlaid. We also compare with the independent implementation of chapter T2 (code/chT2/lib_lisa.py).
Writes: figures/ch10/robson_curve.pdf, results/ch10/11_lisa_robson.tex
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "chT2"))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, SERIES
import lib_gw10 as L

setup()
f = np.geomspace(1e-5, 1.0, 4000)
Sn = L.Sn_robson(f, "4yr")
Sc = L.S_conf(f, "4yr")
Pn = Sn * 0 + (L.P_oms(f) + 2 * (1 + np.cos(f / L.FSTAR) ** 2) * L.P_acc(f) / (2 * np.pi * f) ** 4) / L.L_LISA**2

# digitise Robson et al. Fig. 4 from the PDF itself: render page 6, find the plot frame, and read the purple
# (sqrt S_n) and green (sqrt S_c) curves column by column from their pixel colours.
def digitise():
    import subprocess, tempfile
    from PIL import Image
    pdf = pathlib.Path(__file__).resolve().parents[2] / "references" / "Robson2019.pdf"
    cache = sorted((pathlib.Path(__file__).resolve().parents[2] / "data" / "ch10").glob("robson_p-*.png"))
    if cache:                                   # pre-rendered page (the cluster has no poppler)
        im = np.array(Image.open(cache[0]).convert("RGB")).astype(int)
    else:
      with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdftoppm", "-f", "6", "-l", "6", "-r", "200", "-png", str(pdf), tmp + "/p"], check=True)
        im = np.array(Image.open(next(pathlib.Path(tmp).glob("p*.png"))).convert("RGB")).astype(int)
    sub = im[750:1350, 400:1250]
    dark = sub.sum(axis=2) < 200
    y0, y1 = np.where(dark.sum(axis=1) > 500)[0][[0, -1]]          # frame: 1e-14 (top) ... 1e-21 (bottom)
    x0, x1 = np.where(dark.sum(axis=0) > 300)[0][[0, -1]]          # frame: 1e-5 (left) ... 1 (right)
    out = {"n": [], "c": []}
    for x in range(x0 + 3, x1 - 3, 6):
        col = sub[:, x]
        fx = 10 ** (-5 + 5 * (x - x0) / (x1 - x0))
        for key, mask in (("n", (col[:, 0] > 100) & (col[:, 1] < 90) & (col[:, 2] > 170)),
                          ("c", (col[:, 0] < 80) & (col[:, 1] > 110) & (col[:, 2] < 170) & (col[:, 2] > 60))):
            ys = np.where(mask)[0]
            ys = ys[(ys > y0 + 2) & (ys < y1 - 2)]
            if ys.size and np.ptp(ys) < 12:                          # one curve in this column (skip the legend)
                out[key].append((fx, 10 ** (-14 - 7 * (ys.mean() - y0) / (y1 - y0))))
    return np.array(out["n"]), np.array(out["c"])


dig_n, dig_c = digitise()
dig_n = dig_n[dig_n[:, 0] < 0.5]
dig_c = dig_c[dig_c[:, 0] < 0.05]                                    # legend region excluded above 0.5 Hz
read_f = np.array([1e-4, 3e-4, 1e-3, 2e-3, 1e-2, 3e-2, 1e-1])
read_Sn = np.exp(np.interp(np.log(read_f), np.log(dig_n[:, 0]), np.log(dig_n[:, 1])))
ours_at = np.sqrt(L.Sn_robson(read_f, "4yr"))
ratio = ours_at / read_Sn
smooth = dig_n[:, 0] < 0.03                                          # below the wiggles of the exact response
dev_lo = np.max(np.abs(np.sqrt(L.Sn_robson(dig_n[smooth, 0], "4yr")) / dig_n[smooth, 1] - 1))

nums = dict(TenBrobFstar=1e3 * L.FSTAR)
for k, (fr, pr, o) in enumerate(zip(read_f, read_Sn, ours_at)):
    nm = "ABCDEFG"[k]
    nums[f"TenBrobRead{nm}"] = pr
    nums[f"TenBrobOurs{nm}"] = o
nums["TenBrobRatioMin"], nums["TenBrobRatioMax"] = ratio.min(), ratio.max()
nums["TenBrobDevLow"] = 100 * dev_lo
i0 = np.argmin(Sn[(f > 1e-3) & (f < 0.05)])
fm = f[(f > 1e-3) & (f < 0.05)]
nums["TenBrobASDmin"], nums["TenBrobfmin"] = np.sqrt(Sn[(f > 1e-3) & (f < 0.05)][i0]), 1e3 * fm[i0]

# SDSS J0651+2844
m1, m2, DL, fin, Tm = 0.5, 0.25, 1e3 * L.PC, 2.6e-3, 4 * L.YEAR
Mc = L.chirp_mass(m1, m2)
hGB = 8 * Tm**0.5 * (Mc * L.TSUN) ** (5 / 3) * np.pi ** (2 / 3) * fin ** (2 / 3) / (5**0.5 * DL / L.c)
nums.update(TenBrobJhGB=hGB, TenBrobJsnr=hGB / np.sqrt(L.Sn_robson(np.array([fin]), "4yr")[0]),
            TenBrobJsnrNoConf=hGB / np.sqrt(L.Sn_robson(np.array([fin]), None)[0]))
# GW150914-like
McG = L.chirp_mass(36.0, 29.0)
nums.update(TenBrobGWfFive=1e3 * L.f_of_tau(5 * L.YEAR, McG), TenBrobGWfOne=1e3 * L.f_of_tau(1 * L.YEAR, McG))
# 1e6 Msun equal mass at z = 3, one year before merger, detector-frame chirp mass
McZ = L.chirp_mass(0.5e6, 0.5e6) * 4.0
nums["TenBrobBigf"] = L.f_of_tau(L.YEAR, McZ)
# chapter T2 cross-check
try:
    import lib_lisa as T
    nums["TenBrobTtwoDiff"] = np.max(np.abs(T.Sn_lisa(f, confusion=True) / Sn - 1))
except Exception as e:  # pragma: no cover
    print("chT2 library not available:", e)
save_numbers("ch10", "11_lisa_robson", L.tidy(nums))
print(nums)

fig, ax = plt.subplots(figsize=(6.0, 3.8))
ax.loglog(f, np.sqrt(Pn), color=SERIES[3], lw=1.0, label=r"$\sqrt{P_n}$ (noise in one channel)")
ax.loglog(f, np.sqrt(L.Sn_instr(f)), color=SERIES[0], lw=1.0, ls="--", label=r"$\sqrt{S_n}$ instrument only")
ax.loglog(f, np.sqrt(Sn), color=SERIES[6], lw=2.0, label=r"$\sqrt{S_n}$ with 4-yr confusion (ours)")
ax.loglog(f, np.sqrt(Sc), color=SERIES[2], lw=1.2, label=r"$\sqrt{S_c}$ (ours)")
ax.plot(dig_n[::3, 0], dig_n[::3, 1], "o", mfc="none", mec="k", ms=4, label="digitised from Robson et al. Fig. 4")
ax.plot(dig_c[::3, 0], dig_c[::3, 1], "s", mfc="none", mec=SERIES[2], ms=3.5)
ax.axvline(L.FSTAR, color="0.5", ls=":", lw=1)
ax.set_ylim(1e-21, 1e-14); ax.set_xlim(1e-5, 1)
ax.set_xlabel("$f$ [Hz]"); ax.set_ylabel(r"spectral density [Hz$^{-1/2}$]")
ax.legend(fontsize=7.5, loc="upper right", framealpha=1.0, edgecolor="none").set_zorder(10)
savefig(fig, "ch10", "robson_curve")
