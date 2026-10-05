"""Fiducial CMB spectra shared by every chapter (Planck 2018 TT,TE,EE+lowE+lensing best fit).

Run once:  python3 code/camb_fiducial.py      -> data/camb_fiducial.npz
Load with: from camb_fiducial import load_fiducial;  ell, clTT = load_fiducial()
Spectra are raw C_ell (not D_ell) in muK^2, lensed, for ell = 0..LMAX (ell=0,1 set to zero).
"""
import pathlib
import numpy as np

NOTES = pathlib.Path(__file__).resolve().parents[1]
PATH = NOTES / "data" / "camb_fiducial.npz"
LMAX = 3000
FIDUCIAL = dict(H0=67.36, ombh2=0.02237, omch2=0.1200, tau=0.0544, As=2.100e-9, ns=0.9649, mnu=0.06)


def compute(params=FIDUCIAL, lmax=LMAX):
    import camb
    p = camb.set_params(**params, lmax=lmax + 200, lens_potential_accuracy=1)
    r = camb.get_results(p)
    cls = r.get_cmb_power_spectra(p, CMB_unit="muK", raw_cl=True)
    tot = cls["total"][: lmax + 1]
    return np.arange(lmax + 1), tot  # columns TT, EE, BB, TE


def load_fiducial(spec="TT"):
    if not PATH.exists():
        main()
    z = np.load(PATH)
    return z["ell"], z[spec]


def main():
    ell, tot = compute()
    PATH.parent.mkdir(exist_ok=True)
    np.savez(PATH, ell=ell, TT=tot[:, 0], EE=tot[:, 1], BB=tot[:, 2], TE=tot[:, 3],
             **{f"param_{k}": v for k, v in FIDUCIAL.items()})
    print(f"wrote {PATH}")


if __name__ == "__main__":
    main()
