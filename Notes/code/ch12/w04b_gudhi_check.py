"""Independent check of the Betti curves of w04_beyond.py with gudhi.

Question: do the Betti and persistent Betti numbers that lib_wl's union-find gives for the
fiducial maps of w04 agree with those of a completely different code?

Computes: the same fiducial maps as w04_beyond.simulate (same seeds, same order of random
draws), their superlevel persistence with gudhi's PeriodicCubicalComplex (pixels are the top
cells, so corner-touching pixels connect, as in lib_wl), the two wrap-around loops of the torus
dropped; then the Betti and persistent Betti numbers exactly as w04 does.

Writes: data/ch12/w_beyond_gudhi_ref.npz (read by w04_beyond.py, which reports the largest
difference between the two codes).
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import gudhi

from common import rng_for, DATA
import lib_wl as wl
from w04_beyond import NFID, NUS, PB_GRID, model, observe

OUT = DATA / "ch12" / "w_beyond_gudhi_ref.npz"


def gudhi_pairs(u):
    """(H0, H1) superlevel pairs (b, d), b >= d, of a periodic map, by gudhi."""
    cc = gudhi.PeriodicCubicalComplex(top_dimensional_cells=-u, periodic_dimensions=[True, True])
    cc.compute_persistence()
    h0 = np.array([(-b, -d) for b, d in cc.persistence_intervals_in_dimension(0)], float).reshape(-1, 2)
    h1 = np.array([(-b, -d) for b, d in cc.persistence_intervals_in_dimension(1) if np.isfinite(d)],
                  float).reshape(-1, 2)
    return h0, h1                      # the immortal island has d = -inf, as in lib_wl


def main():
    t0 = time.time()
    sp = dict(np.load(DATA / "ch12" / "w_cl.npz"))
    sig_n, _ = wl.sigma_noise_smoothed()
    s_pix = wl.sigma_pix()
    out = {}
    for kind in ("ln", "g"):
        rng = rng_for("ch12", f"w04_{kind}")          # the stream of w04_beyond.simulate
        fid = model(sp, "fid", kind)
        betti, pbetti = [], []
        for _ in range(NFID[kind]):                     # the fiducial maps come first in that stream
            w = rng.standard_normal((wl.N, wl.N))
            n = rng.standard_normal((2, wl.N, wl.N))
            u = wl.smooth(observe(fid(w), n, s_pix), wl.THETA_G) / sig_n
            h0, h1 = gudhi_pairs(u)
            betti.append(np.concatenate([wl.betti_from_pairs(h0, NUS), wl.betti_from_pairs(h1, NUS)]))
            pbetti.append(np.concatenate([wl.persistent_betti(h0, PB_GRID), wl.persistent_betti(h1, PB_GRID)]))
        out[f"{kind}_fid_betti"] = np.array(betti, float)
        out[f"{kind}_fid_pbetti"] = np.array(pbetti, float)
        print(kind, NFID[kind], "maps done", round(time.time() - t0), "s")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, **out)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
