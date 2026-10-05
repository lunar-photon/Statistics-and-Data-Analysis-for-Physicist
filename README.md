# Statistics and Data Analysis for the Physicist

A self-contained book on statistics and data analysis for physicists, from probability and estimation through
Bayesian inference, Monte Carlo, Fisher forecasts and sampling, to analyses of real data: the Planck CMB maps,
the BOSS CMASS galaxy survey and the ATLAS H→γγ open data.

By Chandra Prakash, written with Claude.

## What is here

| Folder | Contents |
|---|---|
| `Notes/` | The book: `Statistics_and_Data_Analysis_for_the_Physicist.pdf`, its LaTeX source, and the Python scripts behind every figure and number |
| `Python cheatsheet/` | A short cheat sheet of the Python the book's code uses (`.tex` and `.pdf`) |

Inside `Notes/`:

- `main.tex`, `notes.sty`, `references.bib`: the book's main file, style and bibliography.
- `chapters/`: one `.tex` file per chapter.
- `code/chNN/`: the scripts for each chapter. Each starts with a docstring saying what question it answers, what it computes and what it writes.
- `figures/`: the figures the scripts produced.
- `results/`: the few script outputs that the book prints verbatim.

## Building the book

```bash
cd Notes
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Running the code

Each script runs on its own from the `Notes/` folder, for example

```bash
python3 code/ch02/02_binomial.py
```

and writes its figure to `figures/chNN/` and its numbers to `results/chNN/` (as LaTeX macros the text reads).
Random numbers are seeded from the script's name (`rng_for` in `code/common.py`), so every run reproduces the
book's numbers.

Python packages used: `numpy`, `scipy`, `matplotlib`, `sympy`, `mpmath`, `numba`, `healpy`, `astropy`, `camb`,
`iminuit`, `emcee`, `getdist`, `uproot`, `awkward`, `gudhi`, `ripser`, `persim`, `scikit-image`, `pillow`,
`pymupdf`. Most chapters need only the first three.

Some scripts are heavy (thousands of simulated skies) and cache their simulations in `data/`, which is not part
of this repository; they rebuild it on the first run.

## Data

The real-data analyses read public data sets, which are not included here:

- Planck PR3 SMICA maps and common mask: <https://irsa.ipac.caltech.edu/data/Planck/release_3/>
- SDSS-III BOSS DR12 large-scale-structure catalogues (CMASS): <https://data.sdss.org/sas/dr12/boss/lss/>
- ATLAS Open Data, H→γγ (2020 release): <https://opendata.cern.ch/eos/opendata/atlas/OutreachDatasets/2020-08-19/GamGam/>

The script that reads each data set says which files it expects and where to put them.
