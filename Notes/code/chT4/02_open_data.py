"""02_open_data.py -- what is in one row of the 13 TeV ATLAS Open Data diphoton sample.

Question: the diphoton capstone reads ROOT files of reconstructed events. What does each column
mean, how is the diphoton mass built from them, how many events survive a simple photon selection,
and how many Higgs-boson events does the Standard Model predict for the released 10 fb^-1?

Computes:
  (a) the list of branches (name and type) of the data tree, written to results/chT4/02_branches.txt;
  (b) the photon columns of the first selected event, and m_gg from them by hand, two ways;
  (c) an ATLAS-2012-like selection (trigger, two tight photons, |eta| < 2.37 outside 1.37-1.52,
      E_T > 40 / 30 GeV, calorimeter isolation < 4 GeV) on all data files present, with the
      number of events left after each requirement (the cut flow);
  (d) the same selection on the gluon-fusion H -> gamma gamma simulation: its event weights
      (mcWeight x scale factors), the expected number of selected signal events
      sigma B L x (sum of selected weights)/(sum of all weights), the weighted cut flow, the width
      of the simulated peak, and the cross section and weight sum stored in the file itself;
  (e) what the columns look like: photon eta, pT, isolation and multiplicity in data and signal;
      the fraction of converted photons and the size of the release's photon_pt_syst column;
  (f) an exponential through the sidebands of the data spectrum (an illustration of a
      sideband template, not a fit with uncertainties).
Reads: data/ch13/atlas_gamgam/*.root (downloaded for the diphoton capstone; fetched into
       data/chT4/ if absent).
Writes: figures/chT4/opendata_mgg.pdf, figures/chT4/opendata_columns.pdf,
        results/chT4/02_open_data.tex, results/chT4/02_branches.txt
"""
import sys, pathlib, urllib.request
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
import uproot
import awkward as ak
from common import setup, savefig, save_numbers, SERIES, NOTES, RES

setup()
URL = "https://opendata.cern.ch/eos/opendata/atlas/OutreachDatasets/2020-08-19/GamGam"
SRC = NOTES / "data" / "ch13" / "atlas_gamgam"
OWN = NOTES / "data" / "chT4"
DATA_FILES = [f"data_{p}.GamGam.root" for p in "ABCD"]
MC_FILE = "mc_343981.ggH125_gamgam.GamGam.root"
# normalisation of the gluon-fusion sample, from the release's infofile (sigma x B in pb, sum of
# weights); the same two numbers are also stored in every row of the file (checked below)
XSEC_PB, SUMW, LUMI_PB = 0.102, 55922617.6297, 10_000.0


def path_of(name, needed=True):
    for d in (SRC, OWN):
        if (d / name).exists():
            return d / name
    if not needed:
        return None
    OWN.mkdir(parents=True, exist_ok=True)
    sub = "MC" if name.startswith("mc_") else "Data"
    print(f"[download] {name}")
    urllib.request.urlretrieve(f"{URL}/{sub}/{name}", OWN / name)
    return OWN / name


def tree_of(path):
    f = uproot.open(path)
    name = [k for k, c in f.classnames().items() if c == "TTree"][0]
    return f[name.split(";")[0]]


# ---- (a) the columns
t0 = tree_of(path_of(DATA_FILES[0]))
lines = [f"tree: {t0.name}, entries in {DATA_FILES[0]}: {t0.num_entries}"]
lines += [f"{b.name:28s} {b.typename}" for b in t0.branches]
(RES / "chT4").mkdir(parents=True, exist_ok=True)
(RES / "chT4" / "02_branches.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

PH = ["photon_n", "photon_pt", "photon_eta", "photon_phi", "photon_E", "photon_isTightID",
      "photon_ptcone30", "photon_etcone20", "photon_convType", "photon_pt_syst", "trigP"]
MCW = ["mcWeight", "scaleFactor_PHOTON", "scaleFactor_PhotonTRIGGER", "scaleFactor_PILEUP"]
CUTS = ["in file", "photon trigger", "two photons", "both tight", "both in calorimeter",
        "pT > 40, 30 GeV", "both isolated", "100 < m < 160 GeV"]


def first2(x, fill=0.0):
    """The first two photons of every event as an (events, 2) array (events with fewer are padded)."""
    return ak.to_numpy(ak.fill_none(ak.pad_none(x, 2, clip=True), fill))


def select(a):
    """ATLAS-2012-like diphoton selection. Returns the per-cut masks, m_gg [GeV] and photon arrays."""
    n = ak.to_numpy(a["photon_n"])
    ok = n >= 2
    pt2, eta2, phi2 = first2(a["photon_pt"]), first2(a["photon_eta"]), first2(a["photon_phi"])
    scale = 1000.0 if np.median(pt2[ok, 0]) > 1000 else 1.0     # the 2020 release stores MeV
    pt2 = pt2 / scale
    tight2 = first2(a["photon_isTightID"], False).astype(bool)
    iso2 = first2(a["photon_etcone20"]) / scale
    aeta = np.abs(eta2)
    incalo = (aeta < 2.37) & ~((aeta > 1.37) & (aeta < 1.52))
    m = np.sqrt(np.clip(2 * pt2[:, 0] * pt2[:, 1] * (np.cosh(eta2[:, 0] - eta2[:, 1])
                                                    - np.cos(phi2[:, 0] - phi2[:, 1])), 0, None))
    steps = [np.ones(len(n), bool), ak.to_numpy(a["trigP"]).astype(bool), ok, tight2.all(1),
             incalo.all(1), (pt2[:, 0] > 40) & (pt2[:, 1] > 30), (iso2 < 4.0).all(1),
             (m >= 100) & (m < 160)]
    masks = np.cumprod(np.array(steps), axis=0).astype(bool)        # cut i includes cuts 0..i
    return masks, m, pt2, eta2, phi2, iso2, scale


def read(path, extra=()):
    t = tree_of(path)
    extra = [k for k in extra if k in t.keys()]
    return t.arrays(PH + extra, library="ak"), t.num_entries


# ---- (b), (c) and (e) data
mbins = np.arange(100, 161, 1.0)
counts = np.zeros(len(mbins) - 1)
flow_data = np.zeros(len(CUTS))
n_total = 0
used = []
first = None
etab = np.linspace(-2.7, 2.7, 109)
ptb = np.linspace(20, 200, 91)
isob = np.linspace(-5, 15, 81)
h_eta, h_pt1, h_pt2, h_iso = (np.zeros(len(b) - 1) for b in (etab, ptb, ptb, isob))
h_n = np.zeros(6)
conv_sel = conv_all = 0
iso_pass = iso_tot = 0
syst_raw = []
for name in DATA_FILES:
    p = path_of(name, needed=(name == DATA_FILES[0]))
    if p is None:
        continue
    a, ntot = read(p)
    masks, m, pt2, eta2, phi2, iso2, scale = select(a)
    win = masks[-1]
    flow_data += masks.sum(axis=1)
    if first is None:
        i = int(np.flatnonzero(win)[0])
        E = ak.to_numpy(a["photon_E"][i][:2]) / scale
        px, py = pt2[i] * np.cos(phi2[i]), pt2[i] * np.sin(phi2[i])
        pz = pt2[i] * np.sinh(eta2[i])
        m_vec = np.sqrt(max(E.sum()**2 - px.sum()**2 - py.sum()**2 - pz.sum()**2, 0))
        first = dict(scale=scale, pt=pt2[i], eta=eta2[i], phi=phi2[i], E=E, m=m[i], mvec=m_vec,
                     Ecalc=pt2[i] * np.cosh(eta2[i]))
    counts += np.histogram(m[win], bins=mbins)[0]
    # what the columns look like: events with a trigger and two photons (before the other cuts)
    base = masks[2]
    h_eta += np.histogram(eta2[base].ravel(), bins=etab)[0]
    h_pt1 += np.histogram(pt2[base, 0], bins=ptb)[0]
    h_pt2 += np.histogram(pt2[base, 1], bins=ptb)[0]
    h_iso += np.histogram(iso2[base].ravel(), bins=isob)[0]
    iso_pass += int((iso2[base] < 4.0).sum()); iso_tot += int(base.sum()) * 2
    nph = ak.to_numpy(a["photon_n"])[masks[1]]
    h_n += np.bincount(np.clip(nph, 0, 5), minlength=6)[:6]
    conv = first2(a["photon_convType"], 0)
    conv_sel += int((conv[win] != 0).sum()); conv_all += int(win.sum()) * 2
    syst = first2(a["photon_pt_syst"]) / scale
    syst_raw.append(syst[win].ravel())
    n_total += ntot; used.append(name[5])
n_sel = int(flow_data[-1])
syst_raw = np.concatenate(syst_raw)
print(f"photon_pt_syst in data (divided by the unit scale): median {np.median(syst_raw):.4g} "
      f"(the column is filled only in the simulation)")
print(f"data files used: {used}; entries {n_total}; selected in 100-160 GeV: {n_sel}")
print("data cut flow:", dict(zip(CUTS, flow_data.astype(int))))

# ---- (d) signal simulation
mc_tree = tree_of(path_of(MC_FILE))
xs_tree = float(mc_tree["XSection"].array(entry_stop=1, library="np")[0])
sw_tree = float(mc_tree["SumWeights"].array(entry_stop=1, library="np")[0])
print(f"file stores XSection = {xs_tree} pb, SumWeights = {sw_tree}; infofile {XSEC_PB}, {SUMW}")
a, n_mc = read(path_of(MC_FILE), MCW)
masks, m_mc, pt_mc, eta_mc, phi_mc, iso_mc, _ = select(a)
used_w = [k for k in MCW if k in a.fields]
print(f"MC weight = product of {used_w}")
w = np.prod([ak.to_numpy(a[k]).astype(float) for k in used_w], axis=0)
norm = XSEC_PB * LUMI_PB / SUMW                       # expected events per unit weight
flow_mc = np.array([w[mk].sum() for mk in masks]) * norm
win = masks[-1]
effA = w[win].sum() / SUMW
n_exp = XSEC_PB * LUMI_PB * effA
n_prod = XSEC_PB * LUMI_PB
n_mcstat = np.sqrt((w[win]**2).sum()) * norm          # Monte Carlo statistical error of n_exp
q16, q50, q84 = np.percentile(m_mc[win], [16, 50, 84])        # unweighted, shape only
sig68 = 0.5 * (q84 - q16)
in2 = win & (np.abs(m_mc - q50) < 2 * sig68)
frac2 = w[in2].sum() / w[win].sum()
low_tail = w[win & (m_mc < q50 - 3 * sig68)].sum() / w[win].sum()
high_tail = w[win & (m_mc > q50 + 3 * sig68)].sum() / w[win].sum()
mc_hist = np.histogram(m_mc[win], bins=mbins, weights=w[win])[0] * norm
syst_mc = first2(a["photon_pt_syst"]) / 1000.0                  # MeV -> GeV, like photon_pt
syst_rel = (syst_mc[win] / pt_mc[win]).ravel()
print(f"photon_pt_syst in the signal simulation: median {np.median(syst_mc[win]):.4g} GeV, "
      f"median |syst|/pT {np.median(np.abs(syst_rel)):.4g}")
print("signal cut flow:", dict(zip(CUTS, np.round(flow_mc, 1))))

# ---- (f) sideband exponential (least squares on log counts outside 120-130 GeV), illustration only
cen = 0.5 * (mbins[1:] + mbins[:-1])
side = ((cen < 120) | (cen > 130)) & (counts > 0)
slope, icept = np.polyfit(cen[side] - 100, np.log(counts[side]), 1)
bkg = np.exp(icept + slope * (cen - 100))
in_win = (cen > 120) & (cen < 130)
b_win, n_win = bkg[in_win].sum(), counts[in_win].sum()
s_win = mc_hist[in_win].sum()

fig, ax = plt.subplots(1, 2, figsize=(10.0, 3.6), gridspec_kw=dict(width_ratios=[1.5, 1]))
ax[0].errorbar(cen, counts, yerr=np.sqrt(counts), fmt="o", color="k", ms=2.5, lw=0.8,
               label=f"data, {len(used)} of 4 periods")
ax[0].plot(cen, bkg, color=SERIES[0], label="exponential through the sidebands")
ax[0].fill_between(cen, 0, mc_hist, step="mid", color=SERIES[1], alpha=0.5,
                   label=r"SM $gg\to H\to\gamma\gamma$ expected, 10 fb$^{-1}$")
ax[0].axvspan(120, 130, color="0.85", zorder=0)
ax[0].set_xlabel(r"$m_{\gamma\gamma}$ [GeV]"); ax[0].set_ylabel("events / GeV")
ax[0].set_title("(a) the diphoton spectrum and the expected signal")
ax[0].legend(fontsize=7.5, loc="upper right")
hb = np.arange(105, 145.01, 0.5)
ax[1].hist(m_mc[win], bins=hb, weights=w[win] / w[win].sum() / 0.5, histtype="stepfilled",
           color=SERIES[1], alpha=0.5, label="simulated signal")
xs = np.linspace(105, 145, 400)
ax[1].plot(xs, np.exp(-0.5 * ((xs - q50) / sig68)**2) / (np.sqrt(2 * np.pi) * sig68), "k--",
           lw=1.2, label="Gaussian, same 68% width")
ax[1].set_yscale("log"); ax[1].set_ylim(1e-4, 1)
ax[1].set_xlabel(r"$m_{\gamma\gamma}$ [GeV]"); ax[1].set_ylabel("density [1/GeV]")
ax[1].set_title("(b) the signal template has a low-side tail")
ax[1].legend(fontsize=7.5, loc="upper left")
fig.tight_layout()
savefig(fig, "chT4", "opendata_mgg")

# ---- (e) figure: what the columns look like (data after trigger + two photons; signal shapes)
base_mc = masks[2]
fig, ax = plt.subplots(2, 2, figsize=(9.6, 6.4))
c = 0.5 * (etab[1:] + etab[:-1])
ax[0, 0].step(c, h_eta / h_eta.sum() / np.diff(etab), where="mid", color="k", label="data")
ax[0, 0].hist(eta_mc[base_mc].ravel(), bins=etab, density=True, histtype="stepfilled",
              color=SERIES[1], alpha=0.4, label="signal simulation")
for x0 in (-2.37, -1.52, -1.37, 1.37, 1.52, 2.37):
    ax[0, 0].axvline(x0, color="0.5", ls=":", lw=0.8)
ax[0, 0].set_xlabel(r"photon_eta  $\eta$"); ax[0, 0].set_ylabel("density")
ax[0, 0].set_title("(a) where the photons hit: the crack and the edge")
ax[0, 0].set_ylim(0, ax[0, 0].get_ylim()[1] * 1.3)   # head-room for the legend
ax[0, 0].legend(fontsize=7.5, loc="upper center")
c = 0.5 * (ptb[1:] + ptb[:-1])
ax[0, 1].step(c, h_pt1 / h_pt1.sum() / np.diff(ptb), where="mid", color="k", label="data, leading")
ax[0, 1].step(c, h_pt2 / h_pt2.sum() / np.diff(ptb), where="mid", color="k", ls="--",
              label="data, sub-leading")
ax[0, 1].hist(pt_mc[base_mc, 0], bins=ptb, density=True, histtype="stepfilled", color=SERIES[1],
              alpha=0.4, label="signal, leading")
ax[0, 1].set_yscale("log"); ax[0, 1].set_xlabel(r"photon_pt  $p_T$ [GeV]")
ax[0, 1].set_ylabel("density [1/GeV]"); ax[0, 1].set_title("(b) transverse momenta")
ax[0, 1].legend(fontsize=7.5)
c = 0.5 * (isob[1:] + isob[:-1])
ax[1, 0].step(c, h_iso / h_iso.sum() / np.diff(isob), where="mid", color="k", label="data")
ax[1, 0].hist(iso_mc[base_mc].ravel(), bins=isob, density=True, histtype="stepfilled",
              color=SERIES[1], alpha=0.4, label="signal simulation")
ax[1, 0].axvline(4.0, color=SERIES[3], lw=1.2, label="cut: 4 GeV")
ax[1, 0].set_xlabel(r"photon_etcone20 [GeV]"); ax[1, 0].set_ylabel("density [1/GeV]")
ax[1, 0].set_title("(c) energy around the photon (isolation)"); ax[1, 0].legend(fontsize=7.5, loc="upper right")
nmc = np.bincount(np.clip(ak.to_numpy(a["photon_n"])[masks[1]], 0, 5), minlength=6)[:6]
k = np.arange(6)
ax[1, 1].bar(k - 0.18, h_n / h_n.sum(), width=0.36, color="0.3", label="data")
ax[1, 1].bar(k + 0.18, nmc / nmc.sum(), width=0.36, color=SERIES[1], alpha=0.6, label="signal")
ax[1, 1].set_xticks(k); ax[1, 1].set_xticklabels(["0", "1", "2", "3", "4", "5+"])
ax[1, 1].set_xlabel("photon_n"); ax[1, 1].set_ylabel("fraction of triggered events")
ax[1, 1].set_title("(d) photons per event"); ax[1, 1].legend(fontsize=7.5)
fig.tight_layout()
savefig(fig, "chT4", "opendata_columns")

iso_data_pass = iso_pass / iso_tot
iso_mc_pass = (iso_mc[base_mc] < 4.0).mean()
crack_frac = h_eta[(np.abs(0.5 * (etab[1:] + etab[:-1])) > 1.37) &
                   (np.abs(0.5 * (etab[1:] + etab[:-1])) < 1.52)].sum() / h_eta.sum()

f = first
vals = {
    "TfdNfiles": len(used), "TfdNtot": int(n_total), "TfdNsel": n_sel,
    "TfdPtA": f["pt"][0], "TfdPtB": f["pt"][1], "TfdEtaA": f["eta"][0], "TfdEtaB": f["eta"][1],
    "TfdPhiA": f["phi"][0], "TfdPhiB": f["phi"][1], "TfdEA": f["E"][0], "TfdEB": f["E"][1],
    "TfdEcalcA": f["Ecalc"][0], "TfdEcalcB": f["Ecalc"][1], "TfdMfirst": f["m"], "TfdMvec": f["mvec"],
    "TfdNmc": int(n_mc), "TfdEffA": effA, "TfdNexp": round(n_exp, 1), "TfdNprod": int(round(n_prod)),
    "TfdNmcstat": n_mcstat, "TfdXsecTree": xs_tree, "TfdSumwTree": int(round(sw_tree)),
    "TfdSigMC": sig68, "TfdMedMC": q50, "TfdFracTwo": 100 * frac2, "TfdLowTail": 100 * low_tail,
    "TfdHighTail": 100 * high_tail,
    "TfdBwin": int(round(b_win)), "TfdNwin": int(round(n_win)), "TfdSwin": s_win,
    "TfdSlope": slope, "TfdSoverB": 100 * s_win / b_win, "TfdZsimple": s_win / np.sqrt(b_win),
    "TfdIsoData": 100 * iso_data_pass, "TfdIsoMC": 100 * iso_mc_pass, "TfdCrack": 100 * crack_frac,
    "TfdConv": 100 * conv_sel / conv_all, "TfdSystRel": 100 * np.median(np.abs(syst_rel)),
}
letters = "ABCDEFGH"
for j in range(len(CUTS)):
    vals[f"TfdFlowD{letters[j]}"] = int(flow_data[j])
    vals[f"TfdFlowS{letters[j]}"] = round(float(flow_mc[j]), 1)
save_numbers("chT4", "02_open_data", vals)
print(f"first event: {f}")
print(f"MC entries {n_mc}; eff x A = {effA:.4f}; expected {n_exp:.1f} +- {n_mcstat:.1f} (MC stat) of {n_prod:.0f}")
print(f"signal median {q50:.2f}, half 68% width {sig68:.3f}, within 2 sigma {frac2:.3f}, low tail {low_tail:.4f}, high {high_tail:.4f}")
print(f"120-130 GeV window: data {n_win:.0f}, sideband exp {b_win:.1f}, expected signal {s_win:.1f}; slope {slope:.4f}/GeV")
print(f"isolation pass: data {iso_data_pass:.3f}, signal {iso_mc_pass:.3f}; crack fraction {crack_frac:.4f}; "
      f"converted {conv_sel / conv_all:.3f}; median |pt_syst|/pt (signal) {np.median(np.abs(syst_rel)):.4f}")
