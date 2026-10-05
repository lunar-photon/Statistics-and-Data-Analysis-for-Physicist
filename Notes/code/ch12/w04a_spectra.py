"""Convergence spectra and empty-beam shifts for Omega_m and sigma_8 moved by +/-10%.

Question: the Fisher matrices of the peak and topology summaries need derivatives with respect to
Omega_m and sigma_8 estimated from simulated maps.  Counting statistics change in whole units, so a
derivative from a small step is noisy; a larger step (10% instead of 5%) halves that noise.  This
script supplies the Limber C_ell and the shift lambda at the four +/-10% cosmologies.

Writes: data/ch12/w_cl10.npz (keys cl_<name>, lam_<name>, theta_<name>, ells)
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from common import DATA
import lib_wl as wl

STEP = 0.10
OUT = DATA / "ch12" / "w_cl10.npz"


def main():
    ells = dict(np.load(DATA / "ch12" / "w_cl.npz"))["ells"]
    cosmos = {"om_p": (wl.OM_FID * (1 + STEP), wl.S8_FID), "om_m": (wl.OM_FID * (1 - STEP), wl.S8_FID),
              "s8_p": (wl.OM_FID, wl.S8_FID * (1 + STEP)), "s8_m": (wl.OM_FID, wl.S8_FID * (1 - STEP))}
    out = {"ells": ells, "step": np.array(STEP)}
    for name, (om, s8) in cosmos.items():
        cl, lam = wl.limber_and_shift(om, s8, ells)
        out[f"cl_{name}"], out[f"lam_{name}"], out[f"theta_{name}"] = cl, lam, np.array([om, s8])
        print(name, om, s8, "lambda", lam)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, **out)


if __name__ == "__main__":
    main()
