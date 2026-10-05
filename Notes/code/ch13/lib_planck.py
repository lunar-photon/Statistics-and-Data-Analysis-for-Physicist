"""lib_planck.py -- the real Planck sky at laptop scale: files, masks, binning, MASTER.

Shared by the scripts of part 13a (power spectrum of the SMICA map) and usable by 13b:

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import lib_planck as lp
    maps = lp.load_maps(512)            # dict: full, hm1, hm2 (muK, RING, Galactic), beam B_l
    W = lp.load_masks(512)["main"]      # apodised common mask
    edges = lp.planck_edges(lmax=1019)  # Planck's Delta l = 30 bins from l = 30
    P, Q, leff = lp.planck_binning(edges, lmax)

Conventions: raw C_l in muK^2; D_l = l(l+1) C_l / 2 pi; HEALPix RING ordering, Galactic
coordinates (as in the Planck files); angles in radians unless the name says _deg.
Products are cached in data/planck/ (degraded maps and masks, made by 01_planck_data.py) and
data/ch13/ (coupling matrices, simulations).
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import healpy as hp

HERE = pathlib.Path(__file__).resolve().parent
NOTES = HERE.parents[1]
PLANCK = NOTES / "data" / "planck"
DATA = NOTES / "data" / "ch13"
RAW = NOTES.parent / "planck_raw"     # the nside-2048 downloads, kept outside data/ (not synced)
sys.path.insert(0, str(NOTES / "code" / "ch08"))
import lib_masks as lm  # noqa: E402  (taper, coupling matrix, 3j symbols)

DEG = np.pi / 180.0
ARCMIN = DEG / 60.0
FILES = {
    "full": "COM_CMB_IQU-smica_2048_R3.00_full.fits",
    "hm1": "COM_CMB_IQU-smica_2048_R3.00_hm1.fits",
    "hm2": "COM_CMB_IQU-smica_2048_R3.00_hm2.fits",
    "mask": "COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits",
    "binned": "COM_PowerSpect_CMB-TT-binned_R3.01.txt",
    "unbinned": "COM_PowerSpect_CMB-TT-full_R3.01.txt",
}
FWHM_SMICA = 5.0          # arcmin: the effective Gaussian beam of the PR3 CMB maps


# ------------------------------------------------------------------ files
def raw_path(kind: str) -> pathlib.Path:
    """The downloaded nside-2048 file: in planck_raw/ next to the Notes folder, else in data/planck/."""
    p = RAW / FILES[kind]
    return p if p.exists() else PLANCK / FILES[kind]


def degraded_path(kind: str, nside: int) -> pathlib.Path:
    return PLANCK / f"smica_{kind}_n{nside}.fits"


def load_maps(nside: int = 512):
    """Degraded SMICA maps in muK (RING) and the transfer function they carry."""
    out = {k: hp.read_map(degraded_path(k, nside), dtype=np.float64) for k in ("full", "hm1", "hm2")}
    z = np.load(PLANCK / f"smica_transfer_n{nside}.npz")
    out["beam"], out["pixwin"] = z["beam"], z["pixwin"]
    return out


def load_planck_binned():
    """Planck 2018 binned TT spectrum (Plik, l >= 30): l_eff, D_b, sigma_b, best-fit D_b."""
    z = np.loadtxt(PLANCK / FILES["binned"])
    return z[:, 0], z[:, 1], 0.5 * (z[:, 2] + z[:, 3]), z[:, 4]


def load_planck_unbinned():
    z = np.loadtxt(PLANCK / FILES["unbinned"])
    return z[:, 0].astype(int), z[:, 1], z[:, 2], z[:, 3]


# ------------------------------------------------------------------ masks
def mask_components(binary, nside, min_area_deg2=20.0):
    """Split the holes of a 0/1 mask into extended (Galactic) and compact (point-source) parts.

    Holes are the connected sets of masked pixels (8 HEALPix neighbours); a hole larger than
    min_area_deg2 is 'extended'.  Returns two 0/1 masks (1 = kept) whose product is `binary`.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    hole = np.where(binary < 0.5)[0]
    index = -np.ones(binary.size, dtype=np.int64)
    index[hole] = np.arange(hole.size)
    nb = hp.get_all_neighbours(nside, hole)                    # (8, nhole), -1 where none
    rows, cols = [], []
    for k in range(8):
        j = nb[k]
        ok = (j >= 0) & (index[np.where(j >= 0, j, 0)] >= 0)
        rows.append(np.arange(hole.size)[ok])
        cols.append(index[j[ok]])
    r, c = np.concatenate(rows), np.concatenate(cols)
    G = coo_matrix((np.ones(r.size), (r, c)), shape=(hole.size, hole.size))
    ncomp, lab = connected_components(G, directed=False)
    area = np.bincount(lab) * hp.nside2pixarea(nside, degrees=True)
    big = area[lab] > min_area_deg2
    gal = np.ones_like(binary)
    ps = np.ones_like(binary)
    gal[hole[big]] = 0.0
    ps[hole[~big]] = 0.0
    return gal, ps, ncomp, int((area > min_area_deg2).sum())


def distance_to_holes(binary, nside, dmax_deg):
    """Angle (rad) from every pixel to the nearest masked pixel; 0 inside holes, capped at dmax."""
    from scipy.spatial import cKDTree
    v = np.array(hp.pix2vec(nside, np.arange(binary.size))).T
    hole = binary < 0.5
    tree = cKDTree(v[hole])
    chord_max = 2 * np.sin(0.5 * dmax_deg * DEG)
    d, _ = tree.query(v[~hole], k=1, distance_upper_bound=chord_max, workers=4)
    d = np.where(np.isfinite(d), 2 * np.arcsin(np.minimum(d, 2.0) / 2), dmax_deg * DEG)
    out = np.zeros(binary.size)
    out[~hole] = d
    return out


def apodise(binary, nside, width_deg):
    """Cosine taper of width `width_deg` from the edge of every hole inwards (0 in the holes)."""
    if width_deg <= 0:
        return binary.astype(float)
    return lm.taper(distance_to_holes(binary, nside, width_deg), width_deg * DEG)


def latitude_taper(nside, side, b_cut_deg=0.0, width_deg=2.0):
    """Keep b > b_cut (side='north'), b < -b_cut ('south') or |b| > b_cut ('both'), cosine edge."""
    theta, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    b = np.pi / 2 - theta
    x = {"north": b, "south": -b, "both": np.abs(b)}[side] - b_cut_deg * DEG
    return lm.taper(x, width_deg * DEG)


def load_masks(nside: int = 512):
    z = np.load(DATA / f"masks_n{nside}.npz")
    return {k: z[k] for k in z.files}


# ------------------------------------------------------------------ Planck binning
def planck_edges(lmin=30, lmax=1019, dl=30):
    """Planck's high-l TT bins: [30, 60), [60, 90), ... ; the last edge is lmax + 1."""
    e = np.arange(lmin, lmax + 2, dl)
    if e[-1] != lmax + 1:
        e = np.append(e, lmax + 1)
    return e


def planck_binning(edges, lmax):
    """Binning as in the Planck 2018 files (Planck 2018 V, eq. 22).

    C_b = sum_l w_bl C_l with w_bl = l(l+1) / sum_bin l(l+1);   l_eff = sum_l w_bl l;
    the file quotes D_b = l_eff (l_eff + 1) C_b / 2 pi.
    P (nb x lmax+1) maps C_l to D_b; Q (lmax+1 x nb) spreads a D_b back as a C_l flat in the bin,
    so that P Q = 1.
    """
    nb = len(edges) - 1
    P = np.zeros((nb, lmax + 1))
    Q = np.zeros((lmax + 1, nb))
    leff = np.zeros(nb)
    for b in range(nb):
        ls = np.arange(edges[b], edges[b + 1])
        w = ls * (ls + 1.0)
        w /= w.sum()
        leff[b] = np.sum(w * ls)
        f = leff[b] * (leff[b] + 1) / (2 * np.pi)
        P[b, ls] = f * w
        Q[ls, b] = 1.0 / f
    return P, Q, leff


# ------------------------------------------------------------------ MASTER
def coupling(W, lmax, name=None, force=False):
    """M_ll' for the window W up to lmax (cached in data/ch13/coupling_<name>.npz)."""
    nside = hp.npix2nside(W.size)
    path = DATA / f"coupling_{name}_n{nside}_l{lmax}.npz" if name else None
    if path is not None and path.exists() and not force:
        return np.load(path)["M"]
    wl = hp.anafast(W, lmax=min(2 * lmax, 3 * nside - 1), iter=0)
    M = lm.coupling_matrix(wl, lmax, lmax)
    if path is not None:
        DATA.mkdir(parents=True, exist_ok=True)
        np.savez(path, M=M)
    return M


class Master:
    """Binned MASTER for one window: D_hat_b = K^-1 P (pseudo C_l), K = P M T^2 Q.

    T2 is the squared transfer function (beam x pixel window).  window() returns the bandpower
    window F_bl = (K^-1 P M T^2)_bl such that <D_hat_b> = sum_l F_bl C_l exactly.
    """

    def __init__(self, M, T2, edges):
        self.lmax = M.shape[0] - 1
        self.P, self.Q, self.leff = planck_binning(edges, self.lmax)
        self.MT = M * T2[None, : self.lmax + 1]
        self.K = self.P @ self.MT @ self.Q
        self.Kinv = np.linalg.inv(self.K)

    def __call__(self, pcl):
        x = np.asarray(pcl)[..., : self.lmax + 1]
        return (self.Kinv @ (self.P @ x.T)).T

    def window(self):
        return self.Kinv @ self.P @ self.MT


def pseudo_cl(Wmap, maps, lmax):
    """alm of W x map for each map (iter=0: the band limit is well below 3 nside)."""
    return [hp.map2alm(Wmap * m, lmax=lmax, iter=0) for m in maps]
