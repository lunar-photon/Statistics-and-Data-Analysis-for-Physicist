"""01_area_diagrams.py -- the shrinking sample space for three inference problems.

Question: how much of the region where the evidence is true belongs to each
hypothesis?  Drawn as boxes (lib_area.box_diagram): each hypothesis is a strip
as wide as its prior, the evidence E covers the fraction P(E | H_i) of strip i,
so each piece of E has area prior x likelihood, and the posterior of H_i is the
share of E that its piece fills.

Computes, to scale:
  * the medical test (prevalence 0.01, sensitivity 0.95, false-positive rate 0.10);
  * Monty Hall (three doors; Monty opens door C);
  * a trigger: is a triggered event signal or background?  (drawn)

Writes: figures/ch05/area_trigger.pdf, results/ch05/01_area_diagrams.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import setup, savefig, save_numbers
from lib_area import box_diagram, posterior

import matplotlib.pyplot as plt

setup()

# ---------- medical test: prevalence 1%, sensitivity 95%, false positives 10% ----------
pD, sens, fpr = 0.01, 0.95, 0.10
joint, pPos, post = posterior([pD, 1 - pD], [sens, fpr])

# ---------- Monty Hall: you chose A, Monty opened C ----------
mh_prior = [1 / 3, 1 / 3, 1 / 3]
mh_like = [0.5, 1.0, 0.0]          # P(Monty opens C | car behind A, B, C)
mh_joint, mh_pE, mh_post = posterior(mh_prior, mh_like)

# ---------- trigger: signal or background? ----------
pS, eS, eB = 0.05, 0.90, 0.02      # signal fraction before the trigger, efficiencies
tr_joint, tr_pE, tr_post = posterior([pS, 1 - pS], [eS, eB])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7))
box_diagram(ax1, [pS, 1 - pS], [eS, eB], ["$S$", "$B$"], "triggered", after=False,
            title="before the trigger")
box_diagram(ax2, [pS, 1 - pS], [eS, eB], ["$S$", "$B$"], "triggered", after=True,
            title="after: the triggered events only",
            inset=dict(x="box", bounds=[0.36, 0.30, 0.58, 0.62]))
fig.subplots_adjust(wspace=0.08)
savefig(fig, "ch05", "area_trigger")

save_numbers("ch05", "01_area_diagrams", {
    "FiveAMedJointD": joint[0], "FiveAMedJointH": joint[1],
    "FiveAMedPpos": pPos, "FiveAMedPost": post[0],
    "FiveAMontyPE": mh_pE, "FiveAMontyPostA": mh_post[0], "FiveAMontyPostB": mh_post[1],
    "FiveATrigJointS": tr_joint[0], "FiveATrigJointB": tr_joint[1],
    "FiveATrigPE": tr_pE, "FiveATrigPost": tr_post[0],
})
