# VoxLat Task 14 ,  things to check before you submit

Reference numbers below are the numbers in the manuscript draft of 17 October 2026.

## 1. References whose details are not yet confirmed from a primary source (40)

These details come from memory or from secondary sources. Open each DOI or the publisher page and check authors, title, volume, pages and DOI:

[1] van Heerden 2022 · [2] Gai 2019 · [3] Al-Ketan & Abu Al-Rub 2019 · [4] Al-Ketan, Rowshan & Abu Al-Rub 2018 · [10] Kawabe et al. (arXiv 2605.06011; year and author list) · [14] Ohtani et al. (arXiv 2510.10622; year) · [15] Whitaker 1986 · [16] Sanchez-Palencia 1980 · [17] Benenati & Brosilow 1962 · [18] Onck, Andrews & Gibson 2001 · [22] Andreassen & Andreasen 2014 · [23] Brinkman 1949 · [24] Neale & Nader 1974 · [26] Deb et al. 2002 · [29] Daulton et al. 2021 (pages) · [30] Balandat et al. 2020 (pages) · [31] von Schnering & Nesper 1991 · [32] Hildebrand & Rüegsegger 1997 · [33] Schröder-Turk, Fogden & Hyde 2006 · [34] Manwart et al. 2002 · [35] Silvester & Wathen 1994 · [36] Roache 1997 · [37] Backus 1962 · [38] Hashin & Shtrikman 1962 · [39] Zick & Homsy 1982 · [40] Rayleigh 1892 · [41] Shah & London 1978 · [42] Sangani & Acrivos 1982 · [43] Owen 1998 · [44] Arsigny et al. 2006 · [45] Rasmussen & Williams 2006 · [47] Kuleshov et al. 2018 · [48] Vaněk, Mandel & Brezina 1996 · [49] VDI Heat Atlas 2010 · [50] Nield & Bejan 2017 · [51] Schmidt 1967 · [52] Gnielinski 1986 · [53] Ament et al. 2023 · [55] Khaderi, Deshpande & Fleck 2014 · [57] Oberkampf & Roy 2010.

Confirmed so far (Crossref, the publisher page, or read in full during Task 8): [5]–[9], [11]–[13], [19]–[21], [25], [27], [28], [46], [54], [56].

## 2. Claims that rest on a citation you should read yourself

| Where | Claim | Check in |
|---|---|---|
| Intro, Table 1 | You 2024, Hassanirad 2027 and Li 2026 are described from their titles and abstracts only (marked † in Table 1). | full texts [6], [7], [25] |
| Table 1 | Savoldi 2026 does a multi-objective optimisation of a *uniform* gyroid cell. | [12] |
| Table 1 | Modrek 2022: conduction-only topology optimization mapped to gyroid. Ohtani 2025: MMA gradient-based, validated with full CFD. Kawabe 2026: phase-field de-homogenization plus compression tests. | [13], [14], [10] |
| Intro, Sec. 2.3 | "we did not find a study that measures the finite-gap error of the closures used in design…" ,  this is the novelty claim. Run a final Scopus / Web of Science search ("TPMS" AND ("finite size" OR "wall effect" OR "number of cells") AND (permeability OR homogeni*)) and read the Al-Safadi 2026 meta-analysis [21]. | databases, [21] |
| Sec. 2.1 | Rathore 2023 channel C_F differs from the periodic value (Task 8: 1.15× for D, 1.64× for G of Gajetti). | [19] |
| Sec. 2.3 | Near-wall porosity close to one within half a diameter in packed beds. | [17] |
| Sec. 2.3 | Size effects in few-cell cellular solids (cut cells carry less load). | [18] |
| Sec. 4.1 | Minimal-surface areas 3.0915 (G) and 3.8377 (D) per unit cell. | [33] |
| Sec. 5.1 | Bending-dominated exponent of about 2 for the gyroid lattice. | [55], [4] |
| Sec. 5.1 | Maskery et al. prefactors C = 0.69 (G) and 0.68 (D) with m = 2 (their Table 3). | [56] |
| Sec. 4.5 | Coefficients of the Gajetti C_F power laws and of Savoldi Eq. 37 (0.0267 φ^0.19). | [11], [12] |
| Sec. 4.6 | Nu = 5.385 for a parallel-plate channel, one wall at uniform flux, other adiabatic. | [41] |
| Sec. 4.7 | Schmidt curvature factor; VDI laminar coil correlation; the Gnielinski transition rule (implemented from the VDI description; check against VDI Heat Atlas G3). | [49], [51], [52] |
| Sec. 6 | Savoldi et al. report higher measured friction factors for printed gyroids. | [12] |

## 3. Assumptions to state or confirm with your supervisor

- 770 W of the 957 W loss enters the jacket (80% share) ,  an assumption of the reference problem.
- σ_allow = 80 MPa (fatigue-limited) and the rib stress-concentration factor 2.5 used for B1.
- End-ring manifolds of the lattice and of B1 are ideal and lie outside the heated length.

## 4. Figures

- Several figures still carry working labels in their legends (e.g. "Task 10 pick" in Fig. 9). The "Task …" super-titles were cropped, but regenerate Figs. 9–12 without "Task" wording in the legends before submission.
- Figs. 9 and 10 show the optimizer's blend design `best_at_cap` (w ≤ 0.24), not the exported pure-gyroid optimum (same ρ* and L fields). The captions say so. Regenerating them with `task12_slots_best_at_cap_gyroid` (STATUS Task 13, open issue 1) would be cleaner.
- Elsevier wants figures as separate files at submission (300 dpi PNG/TIFF is fine; the repo already has 300 dpi PNGs).
- Equations are native Word equations. LibreOffice without its Math module shows them blank; open the .docx in Microsoft Word.

## 5. Front matter and declarations

- Authors are now Zain ul Abiden (corresponding, zabiden.ug23smme@student.nust.edu.pk) and Ahmed Mukhtar Bukhari, both at Department of Aerospace Engineering, SMME, NUST, H-12, Islamabad 44000. Check the official affiliation wording and add ORCIDs in the submission system.
- The CRediT, funding, acknowledgements and generative-AI sections were removed on request. Elsevier's policy asks authors to declare AI-assisted writing; the submission form may ask about it separately, so answer it there.
- Data availability: mint a Zenodo DOI for a tagged release of github.com/zain4366/voxlat and add it.
- Add a `LICENSE` file to the repository (the MIT text is in `repo_additions/LICENSE`). Only `pyproject.toml` mentions MIT at the moment.

## 6. JOSS paper (paper.md) ,  blockers

- JOSS requires **at least six months of public development history** for new repositories; the repo was created on 7 October 2026, so the earliest realistic JOSS submission is about **April 2027**.
- JOSS needs an OSI licence *file* (see above), a README with install and usage instructions (present) and community guidelines (add a short CONTRIBUTING section on how to report issues and contribute).
- The "Research impact statement" must show realized or concrete near-term impact; the main paper (once accepted or posted as a preprint) is the best evidence.
- Replace the ORCID placeholder. The AI-usage disclosure section is mandatory in JOSS, so it stays in paper.md; make sure it is accurate.

## 7. Numbers

Every number in the draft was cross-checked against STATUS.md and `results/tables/T01–T09`. Derived values to recheck:
- 10.1% vs B2 = 21.92/24.37 − 1. The T07 row says −10.05%, which rounds to 10.1%.
- 7–37% Darcy-term underestimate = 1/(1 − a_K/N) − 1 for a_K = 0.20–0.27 and N = 1–3.
- Factor 2.6 = 21.92/8.57 (slots vs end-ring optimum).
- "Inertia carries 62%" of the reference Δp = 1 − 38% Darcy share.
