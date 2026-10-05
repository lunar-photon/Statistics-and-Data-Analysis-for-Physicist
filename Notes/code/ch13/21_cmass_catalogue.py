"""21_cmass_catalogue.py -- from the public BOSS DR12 CMASS files to comoving positions and weights.

Question: where, in comoving coordinates, are the CMASS galaxies of the southern Galactic cap,
what weight does each one carry, and what does the matching random catalogue look like?
Computes: reads galaxy_DR12v5_CMASS_South.fits.gz and random0_DR12v5_CMASS_South.fits.gz
(downloaded once from https://data.sdss.org/sas/dr12/boss/lss/ into build/sdss_raw/, 1.2 GB);
keeps 0.43 < z < 0.7; converts (RA, Dec, z) to comoving Cartesian positions in Mpc/h with the
Planck 2018 fiducial cosmology (comoving distance D_C(z) from CAMB); forms the galaxy weight
w_tot * w_FKP with w_tot = w_systot (w_cp + w_noz - 1) and the random weight w_FKP; keeps a
random subsample of NRAND_RATIO times the number of galaxies; assigns every object to one of
NJK jackknife regions on the sky (k-means on the unit vectors of the randoms).
Writes: data/ch13/cmass_south.npz, figures/ch13/cmass_footprint.pdf, figures/ch13/cmass_nz.pdf,
results/ch13/21_cmass_catalogue.tex
"""
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, SERIES, DATA, NOTES
from camb_fiducial import FIDUCIAL

RAW = NOTES / "build" / "sdss_raw"
URL = "https://data.sdss.org/sas/dr12/boss/lss/"
FILES = {"gal": "galaxy_DR12v5_CMASS_South.fits.gz", "ran": "random0_DR12v5_CMASS_South.fits.gz"}
ZMIN, ZMAX = 0.43, 0.70          # the CMASS redshift range of the BOSS analyses
NRAND_RATIO = 10                 # randoms kept per galaxy (the file holds about 50 per galaxy)
NJK = 120                        # jackknife regions
P0 = 1.0e4                       # (Mpc/h)^3, the P_0 of the catalogue's FKP weights
rng = rng_for("ch13", "21_cmass_catalogue")


def fetch(name):
    path = RAW / name
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-sSL", "-o", str(path), URL + name], check=True)
    return path


def read(name, cols):
    """Selected columns of a FITS table; the .gz is unpacked once so the file can be memory-mapped."""
    from astropy.io import fits
    plain = RAW / name.removesuffix(".gz")
    if not plain.exists():
        with open(plain, "wb") as out:
            subprocess.run(["gzip", "-dc", str(fetch(name))], stdout=out, check=True)
    with fits.open(plain, memmap=True) as h:
        d = h[1].data
        return {c: np.array(d[c], dtype=float) for c in cols}


# ---- 1. the fiducial distance-redshift relation, D_C(z) = c int_0^z dz'/H(z'), from CAMB
def comoving_distance_table():
    import camb
    p = camb.set_params(H0=FIDUCIAL["H0"], ombh2=FIDUCIAL["ombh2"], omch2=FIDUCIAL["omch2"],
                        mnu=FIDUCIAL["mnu"], ns=FIDUCIAL["ns"], As=FIDUCIAL["As"])
    bg = camb.get_background(p)
    zt = np.linspace(0.0, 1.0, 2001)
    h = FIDUCIAL["H0"] / 100
    dc = bg.comoving_radial_distance(zt) * h                  # Mpc -> Mpc/h
    hz = bg.hubble_parameter(zt)                              # km/s/Mpc
    rd = bg.get_derived_params()["rdrag"]                    # Mpc
    return zt, dc, hz, rd, h


zt, dct, hzt, RD_MPC, H = comoving_distance_table()
dc_of_z = lambda z: np.interp(z, zt, dct)


def to_xyz(ra, dec, z):
    """Comoving Cartesian position: distance D_C(z) along the unit vector of (RA, Dec)."""
    r = dc_of_z(z)
    a, d = np.radians(ra), np.radians(dec)
    return np.column_stack([r * np.cos(d) * np.cos(a), r * np.cos(d) * np.sin(a), r * np.sin(d)])


# ---- 2. read, cut, weight
g = read(FILES["gal"], ["RA", "DEC", "Z", "WEIGHT_SYSTOT", "WEIGHT_CP", "WEIGHT_NOZ", "WEIGHT_FKP", "NZ"])
n_file = len(g["Z"])
sel = (g["Z"] > ZMIN) & (g["Z"] < ZMAX)
g = {k: v[sel] for k, v in g.items()}
w_tot = g["WEIGHT_SYSTOT"] * (g["WEIGHT_CP"] + g["WEIGHT_NOZ"] - 1)     # Reid et al. 2016, eq. (50)
w_fkp_check = 1 / (1 + g["NZ"] * P0)                                  # Reid et al. 2016, eq. (53)
wg = w_tot * g["WEIGHT_FKP"]

r = read(FILES["ran"], ["RA", "DEC", "Z", "WEIGHT_FKP", "NZ"])
n_ran_file = len(r["Z"])
rsel = (r["Z"] > ZMIN) & (r["Z"] < ZMAX)
r = {k: v[rsel] for k, v in r.items()}
n_ran_cut = len(r["Z"])
keep = rng.choice(n_ran_cut, size=min(n_ran_cut, NRAND_RATIO * len(g["Z"])), replace=False)
keep.sort()
r = {k: v[keep] for k, v in r.items()}
wr = r["WEIGHT_FKP"]

xg, xr = to_xyz(g["RA"], g["DEC"], g["Z"]), to_xyz(r["RA"], r["DEC"], r["Z"])

# ---- 3. jackknife regions: k-means on the sky positions of (a subsample of) the randoms
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree


def unit(ra, dec):
    a, d = np.radians(ra), np.radians(dec)
    return np.column_stack([np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)])


ur, ug = unit(r["RA"], r["DEC"]), unit(g["RA"], g["DEC"])
sub = ur[rng.choice(len(ur), 200_000, replace=False)]
centres, _ = kmeans2(sub, NJK, minit="++", seed=rng, iter=30)
reg_r = cKDTree(centres).query(ur)[1]
reg_g = cKDTree(centres).query(ug)[1]

# ---- 4. a few numbers
z_eff = np.sum(wg * g["Z"]) / np.sum(wg)                       # weighted mean redshift
f_sky_deg2 = 2525.0                                             # effective area, Reid et al. Table 2
# shell volume for the solid angle Omega = A (pi/180)^2 sr:  Omega (D2^3 - D1^3)/3
omega = f_sky_deg2 * (np.pi / 180) ** 2
vol = omega * (dc_of_z(ZMAX) ** 3 - dc_of_z(ZMIN) ** 3) / 3
nbar_mean = len(g["Z"]) / vol
# V_eff = int dV [nP0/(1+nP0)]^2; each galaxy stands for a volume 1/n(z), so sum over galaxies
veff = np.sum(((g["NZ"] * P0) / (1 + g["NZ"] * P0)) ** 2 / g["NZ"])

counts_jk = np.bincount(reg_g, minlength=NJK)
(DATA / "ch13").mkdir(parents=True, exist_ok=True)
np.savez_compressed(DATA / "ch13" / "cmass_south.npz",
                    xg=xg.astype(np.float32), wg=wg, zg=g["Z"], rag=g["RA"], decg=g["DEC"], nzg=g["NZ"],
                    xr=xr.astype(np.float32), wr=wr, zr=r["Z"], rar=r["RA"], decr=r["DEC"],
                    reg_g=reg_g, reg_r=reg_r, wsys=g["WEIGHT_SYSTOT"], wcp=g["WEIGHT_CP"],
                    wnoz=g["WEIGHT_NOZ"], wfkp=g["WEIGHT_FKP"], z_eff=z_eff, rd_mpc=RD_MPC, h=H,
                    zt=zt, dct=dct, hzt=hzt)

save_numbers("ch13", "21_cmass_catalogue", {
    "CmNfile": f"{n_file:,}".replace(",", "{,}"),
    "CmNgal": f"{len(g['Z']):,}".replace(",", "{,}"),
    "CmNranFile": f"{n_ran_file:,}".replace(",", "{,}"),
    "CmNranCut": f"{n_ran_cut:,}".replace(",", "{,}"),
    "CmNran": f"{len(r['Z']):,}".replace(",", "{,}"),
    "CmRanRatio": NRAND_RATIO,
    "CmZeff": f"{z_eff:.3f}",
    "CmDcMin": f"{dc_of_z(ZMIN):.0f}",
    "CmDcMax": f"{dc_of_z(ZMAX):.0f}",
    "CmDcEff": f"{dc_of_z(z_eff):.0f}",
    "CmVol": f"{vol / 1e9:.2f}",
    "CmNbar": f"{1e4 * nbar_mean:.2f}",
    "CmVeff": f"{veff / 1e9:.2f}",
    "CmNzPeak": f"{1e4 * g['NZ'].max():.2f}",
    "CmNjk": NJK,
    "CmJkMin": int(counts_jk.min()), "CmJkMax": int(counts_jk.max()),
    "CmFracCp": f"{100 * np.mean(g['WEIGHT_CP'] > 1):.1f}",
    "CmFracNoz": f"{100 * np.mean(g['WEIGHT_NOZ'] > 1):.1f}",
    "CmSysLo": f"{np.percentile(g['WEIGHT_SYSTOT'], 2.5):.3f}",
    "CmSysHi": f"{np.percentile(g['WEIGHT_SYSTOT'], 97.5):.3f}",
    "CmSumWtot": f"{np.sum(w_tot):.0f}",
    "CmFkpMaxDiff": f"{np.max(np.abs(w_fkp_check - g['WEIGHT_FKP'])):.1e}",
    "CmFkpLo": f"{g['WEIGHT_FKP'].min():.2f}", "CmFkpHi": f"{g['WEIGHT_FKP'].max():.2f}",
    "CmRdMpc": f"{RD_MPC:.2f}", "CmRdH": f"{RD_MPC * H:.2f}",
})

# ---- 5. figures: footprint with jackknife regions, and n(z) with the weights
setup()
fig, ax = plt.subplots(figsize=(6.4, 3.0))
ra_w = lambda ra: np.where(ra > 180, ra - 360, ra)          # the southern cap straddles RA = 0
idx = rng.choice(len(r["RA"]), 150_000, replace=False)
cmap = plt.get_cmap("tab20")
ax.scatter(ra_w(r["RA"][idx]), r["DEC"][idx], s=0.15, c=cmap(reg_r[idx] % 20), rasterized=True)
gi = rng.choice(len(g["RA"]), 6000, replace=False)
ax.scatter(ra_w(g["RA"][gi]), g["DEC"][gi], s=0.4, c="k", rasterized=True)
ax.set_xlabel("RA [deg] (shifted to $-180\\ldots180$)")
ax.set_ylabel("Dec [deg]")
ax.invert_xaxis()
ax.set_aspect("equal")
savefig(fig, "ch13", "cmass_footprint")

fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.8))
zb = np.linspace(ZMIN, ZMAX, 55)
zc = 0.5 * (zb[1:] + zb[:-1])
shell = omega * (dc_of_z(zb[1:]) ** 3 - dc_of_z(zb[:-1]) ** 3) / 3
n_raw = np.histogram(g["Z"], zb)[0] / shell
n_tot = np.histogram(g["Z"], zb, weights=w_tot)[0] / shell
axs[0].step(zc, 1e4 * n_raw, where="mid", color=SERIES[1], label="galaxies, unweighted")
axs[0].step(zc, 1e4 * n_tot, where="mid", color=SERIES[0], label="weighted by $w_{\\rm tot}$")
order = np.argsort(g["Z"])
axs[0].plot(g["Z"][order], 1e4 * g["NZ"][order], color="k", ls="--", lw=1, label="NZ column")
axs[0].set_xlabel("$z$"); axs[0].set_ylabel(r"$10^4\,\bar n(z)\;[h^3\,{\rm Mpc}^{-3}]$")
axs[0].legend(fontsize=7)
axs[1].plot(g["Z"][order], g["WEIGHT_FKP"][order], color=SERIES[2])
axs[1].set_xlabel("$z$"); axs[1].set_ylabel(r"$w_{\rm FKP}=1/(1+\bar n P_0)$")
savefig(fig, "ch13", "cmass_nz")
print(f"galaxies {len(g['Z'])}, randoms kept {len(r['Z'])} of {n_ran_cut}, z_eff {z_eff:.3f}")
