"""31_hyy_select.py -- from the ATLAS 13 TeV Open Data diphoton files to a list of masses.

Question: which recorded collisions contain two well-measured, isolated photons, what is the
invariant mass of each pair, and how many Higgs-boson decays does the Standard Model predict
among them?
Computes: reads the GamGam ntuples of the 2020 ATLAS Open Data release (record 15006 of
opendata.cern.ch, 10 fb^-1 of 2016 data, periods A-D, plus the simulated ggH, VBF, WH and ZH
signal samples at m_H = 125 GeV) with uproot (the branches of the tree "mini"); applies the
selection of the release's HyyAnalysis example (photon trigger; exactly two tight photons with
pT > 25 GeV and |eta| < 2.37 outside 1.37-1.52; track and calorimeter isolation below 6.5% of pT;
pT/m > 0.35 and 0.25; 105 < m < 160 GeV); computes m = sqrt(2 pT1 pT2 (cosh d_eta - cos d_phi));
weights simulated events by lumi * sigma / sum(w) * mcWeight * scale factors; records the
cut flow, the mass shift from the release's photon_pt_syst, and the "central unconverted"
category of the example.
Inputs: data/ch13/atlas_gamgam/*.root (1.9 GB, downloaded once with curl from
https://opendata.cern.ch/eos/opendata/atlas/OutreachDatasets/2020-08-19/GamGam/).
Writes: data/ch13/hyy_events.npz, figures/ch13/hyy_spectrum.pdf, figures/ch13/hyy_cutflow.pdf,
results/ch13/31_hyy_select.tex
"""
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, SERIES, DATA
import uproot
import awkward as ak

RAW = DATA / "ch13" / "atlas_gamgam"
PERIODS = ["A", "B", "C", "D"]
SIGNALS = {"ggH": "mc_343981.ggH125_gamgam", "VBF": "mc_345041.VBFH125_gamgam",
           "WH": "mc_345318.WpH125J_Wincl_gamgam", "ZH": "mc_345319.ZH125J_Zincl_gamgam"}
LUMI_PB = 10064.0                  # pb^-1, the luminosity the release's plotting code uses
MLO, MHI = 105.0, 160.0            # GeV, the mass window

SCAL = {"trigP": np.bool_, "photon_n": np.uint32}
SCAL_MC = {"mcWeight": np.float32, "scaleFactor_PHOTON": np.float32,
           "scaleFactor_PhotonTRIGGER": np.float32, "scaleFactor_PILEUP": np.float32,
           "XSection": np.float32, "SumWeights": np.float32}
VEC = {"photon_pt": np.float32, "photon_eta": np.float32, "photon_phi": np.float32,
       "photon_isTightID": np.bool_, "photon_ptcone30": np.float32,
       "photon_etcone20": np.float32, "photon_convType": np.int32, "photon_pt_syst": np.float32}


def read_branches(path, scalars, vectors):
    """One value per event for `scalars`; (photons per event, flat photon values) for `vectors`."""
    with uproot.open(path) as f:
        arr = f["mini"].arrays(list(scalars) + list(vectors), library="ak")
    out = {b: ak.to_numpy(arr[b]).astype(t) for b, t in scalars.items()}
    for b, t in vectors.items():
        out[b] = (ak.to_numpy(ak.num(arr[b])).astype(np.int64), ak.to_numpy(ak.flatten(arr[b])).astype(t))
    return out


CUTS = ["all", "trigger", "two good photons", "isolation", "pT/m", "mass window"]


def select(path, is_mc):
    """Apply the HyyAnalysis selection; return masses, weights, category, mass shift, cut flow."""
    b = read_branches(path, {**SCAL, **(SCAL_MC if is_mc else {})}, VEC)
    nev = len(b["trigP"])
    counts = b["photon_pt"][0]
    assert np.array_equal(counts, b["photon_n"].astype(np.int64)), "photon_n disagrees with vector length"
    ev = np.repeat(np.arange(nev), counts)                       # event index of every photon
    v = {k: b[k][1] for k in VEC}
    w = np.ones(nev)
    if is_mc:
        w = (b["mcWeight"] * b["scaleFactor_PHOTON"] * b["scaleFactor_PhotonTRIGGER"]
             * b["scaleFactor_PILEUP"]).astype(float)
        w *= LUMI_PB * float(b["XSection"][0]) / float(b["SumWeights"][0])
    flow = [w.sum()]
    trig = b["trigP"]
    flow.append(w[trig].sum())
    aeta = np.abs(v["photon_eta"])
    good = (v["photon_isTightID"] & (v["photon_pt"] > 25e3) & (aeta < 2.37)
            & ((aeta < 1.37) | (aeta > 1.52)))
    ngood = np.bincount(ev[good], minlength=nev)
    keep = trig & (ngood == 2)
    flow.append(w[keep].sum())
    # flat indices of the two good photons of every kept event (photons are stored by falling pT)
    gidx = np.flatnonzero(good & keep[ev])
    i1, i2 = gidx[0::2], gidx[1::2]
    swap = v["photon_pt"][i2] > v["photon_pt"][i1]              # make photon 1 the leading one
    i1, i2 = np.where(swap, i2, i1), np.where(swap, i1, i2)
    evk = ev[i1]
    assert np.array_equal(evk, ev[i2])
    pt1, pt2 = v["photon_pt"][i1] / 1e3, v["photon_pt"][i2] / 1e3   # MeV -> GeV
    iso = ((v["photon_ptcone30"][i1] / v["photon_pt"][i1] < 0.065)
           & (v["photon_etcone20"][i1] / v["photon_pt"][i1] < 0.065)
           & (v["photon_ptcone30"][i2] / v["photon_pt"][i2] < 0.065)
           & (v["photon_etcone20"][i2] / v["photon_pt"][i2] < 0.065))
    flow.append(w[evk[iso]].sum())
    deta = v["photon_eta"][i1] - v["photon_eta"][i2]
    dphi = np.abs(v["photon_phi"][i1] - v["photon_phi"][i2])
    dphi = np.where(dphi < np.pi, dphi, 2 * np.pi - dphi)
    m = np.sqrt(2 * pt1 * pt2 * (np.cosh(deta) - np.cos(dphi)))
    m = np.maximum(m, 1e-6)                                      # guards pT/m for collinear pairs
    kin = iso & (pt1 / m > 0.35) & (pt2 / m > 0.25)
    flow.append(w[evk[kin]].sum())
    win = kin & (m > MLO) & (m < MHI)
    flow.append(w[evk[win]].sum())
    cat = ((np.abs(v["photon_eta"][i1]) < 0.75) & (np.abs(v["photon_eta"][i2]) < 0.75)
           & (v["photon_convType"][i1] == 0) & (v["photon_convType"][i2] == 0))
    # mass shift if both photon pT move by their quoted systematic: m ~ sqrt(pT1 pT2)
    rel = 0.5 * (v["photon_pt_syst"][i1] / v["photon_pt"][i1] + v["photon_pt_syst"][i2] / v["photon_pt"][i2])
    return dict(m=m[win], w=w[evk[win]], cat=cat[win], rel=rel[win], flow=np.array(flow), nev=nev)


def main():
    setup()
    out, nums = {}, {}
    flows = {}
    for p in PERIODS:
        r = select(RAW / f"data_{p}.GamGam.root", False)
        out[f"data_{p}"] = r["m"]
        out[f"cat_{p}"] = r["cat"]
        flows[f"data_{p}"] = r["flow"]
        print(f"data {p}: {r['nev']} events read, {len(r['m'])} selected")
    for s, f in SIGNALS.items():
        r = select(RAW / f"{f}.GamGam.root", True)
        for k in ("m", "w", "cat", "rel"):
            out[f"{s}_{k}"] = r[k]
        flows[s] = r["flow"]
        print(f"{s}: {r['nev']} simulated events, expected {r['w'].sum():.1f} selected")
    np.savez_compressed(DATA / "ch13" / "hyy_events.npz", **out)

    mdat = np.concatenate([out[f"data_{p}"] for p in PERIODS])
    cdat = np.concatenate([out[f"cat_{p}"] for p in PERIODS])
    fdat = sum(flows[f"data_{p}"] for p in PERIODS)
    fsig = sum(flows[s] for s in SIGNALS)
    msig = np.concatenate([out[f"{s}_m"] for s in SIGNALS])
    wsig = np.concatenate([out[f"{s}_w"] for s in SIGNALS])
    csig = np.concatenate([out[f"{s}_cat"] for s in SIGNALS])
    rsig = np.concatenate([out[f"{s}_rel"] for s in SIGNALS])

    # ---- figure: the spectrum, with the expected signal drawn on its own
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.0, 4.6), sharex=True,
                                 gridspec_kw=dict(height_ratios=[3, 1.3], hspace=0.08))
    edges = np.arange(MLO, MHI + 0.01, 1.0)
    cen = 0.5 * (edges[1:] + edges[:-1])
    n, _ = np.histogram(mdat, edges)
    s, _ = np.histogram(msig, edges, weights=wsig)
    a1.errorbar(cen, n, yerr=np.sqrt(n), fmt="o", ms=2.5, color="k", lw=0.8, label="data, 10 fb$^{-1}$")
    a1.set_ylabel("events / GeV")
    a1.legend()
    a2.step(cen, s, where="mid", color=SERIES[1], label="simulated SM Higgs ($m_H=125$ GeV)")
    a2.set_ylabel("events / GeV")
    a2.set_xlabel(r"$m_{\gamma\gamma}$ [GeV]")
    a2.legend(loc="upper right")
    savefig(fig, "ch13", "hyy_spectrum")

    # ---- figure: cut flow, data and expected signal, as fractions of the trigger count
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    x = np.arange(1, len(CUTS))
    ax.semilogy(x, fdat[1:] / fdat[1], "o-", color=SERIES[0], label="data")
    ax.semilogy(x, fsig[1:] / fsig[1], "s-", color=SERIES[1], label="simulated Higgs")
    ax.set_xticks(x, CUTS[1:], rotation=15)
    ax.set_ylabel("fraction kept")
    ax.legend()
    savefig(fig, "ch13", "hyy_cutflow")

    nums.update(
        HyyNData=len(mdat), HyyNCat=int(cdat.sum()),
        HyyNTrig=int(fdat[1]), HyyNTwo=int(fdat[2]), HyyNIso=int(fdat[3]), HyyNKin=int(fdat[4]),
        HyyNAll=int(fdat[0]),
        HyySigExp=round(wsig.sum(), 1), HyySigCat=round(wsig[csig].sum(), 1),
        HyySigTrig=round(fsig[1], 1), HyySigTwo=round(fsig[2], 1), HyySigIso=round(fsig[3], 1),
        HyySigKin=round(fsig[4], 1), HyySigAll=round(fsig[0], 1),
        HyyEffSig=round(fsig[5] / fsig[0], 3),
        HyyRelSyst=round(float(np.average(rsig, weights=wsig)), 4),
        HyyNA=len(out["data_A"]), HyyNB=len(out["data_B"]), HyyNC=len(out["data_C"]), HyyND=len(out["data_D"]),
    )
    for sname in SIGNALS:
        nums[f"HyySig{sname}"] = round(out[f"{sname}_w"].sum(), 1)
    save_numbers("ch13", "31_hyy_select", nums)
    print(nums)


if __name__ == "__main__":
    main()
