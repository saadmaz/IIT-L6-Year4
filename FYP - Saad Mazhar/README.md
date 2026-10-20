# Tea Sense - Final Year Project

**A Near-Infrared Spectroscopy and Artificial Intelligence Framework for Rapid Estimation of Moisture and Quality Parameters in Black Tea**

Muhammed Saad Mazhar | IIT 20221804 | UoW W2053223
Supervisor: Mr. Ajeevan Sivanandhan
BSc (Hons) Computer Science - IIT / University of Westminster - 2026/27

---

## Aim

Design, develop and evaluate an NIR spectroscopy and AI framework for rapid, non-destructive estimation of black-tea quality parameters, validated experimentally on moisture and assessed for extension to polyphenols, TF:TR, caffeine, sugar and glyphosate.

## Research questions

| ID | Question | Answered in |
|----|----------|-------------|
| RQ1 | How accurately can NIR spectra predict black-tea moisture against oven-drying on unseen samples? | `06 Results/RQ1 Moisture Accuracy` |
| RQ2 | How do preprocessing (SNV, MSC, Savitzky-Golay) and wavelength selection (CARS) affect performance? | `06 Results/RQ2 Preprocessing & CARS` |
| RQ3 | Do ML / neural models improve on the PLSR baseline, and is the gain worth it? | `06 Results/RQ3 Model Comparison` |
| RQ4 | How much time does NIR save, and does it hold up in the field? | `06 Results/RQ4 Time Saving & Field Test` |
| RQ5 | Which further parameters can the framework estimate, according to the evidence? | `02 Literature` (Feasibility Matrix) |

## Folder guide

| Folder | What goes here |
|--------|----------------|
| `01 Project Management` | Master Project Tracker, Risk Register, Weekly Logbook, Supervisor Meeting Log; `Module Guides & Templates/` (PP guide, templates) |
| `02 Literature` | `Papers/` (all PDFs) + Literature Review Matrix, Paper Tracking Matrix, Feasibility Matrix (RQ5) |
| `03 Ethics & Protocol` | Ethics application and approval, supplier consent, risk assessment, oven-drying and scanning SOPs, pilot report |
| `04 Data` | `raw/` spectrometer exports, `reference/` oven-drying results, `processed/` code outputs, `Sample Log.xlsx` |
| `05 Code` | `poc/` proof of concept on the public Tecator NIR dataset, `notebooks/` exploration, `src/` pipeline (preprocessing, CARS, models, evaluation), `app/` CLI + Streamlit prototype |
| `06 Results` | `Experiment Log.xlsx` + one folder per research question (tables, figures, saved models) |
| `07 Writing` | `Proposal/` (PP, PPRS), `Thesis/` (chapters + Writing Tracker), `Presentations/` (slides, demos, viva) |
| `99 Archive` | Superseded versions worth keeping (e.g. `Proposal Drafts/`) |

## Rules that keep the project reproducible

1. **`04 Data/raw/` is read-only.** Never edit or overwrite a raw scan. Fix problems in code.
2. **`04 Data/processed/` is produced only by code** and can always be regenerated from `raw/` + `reference/`.
3. **Every sample has one ID** (e.g. `T001`) used in the Sample Log, spectrum file names and oven sheets.
   Spectrum files: `T001_scan1.csv`, `T001_scan2.csv`, `T001_scan3.csv`.
4. **Every experiment run gets a row in `06 Results/Experiment Log.xlsx`** (ID, date, data split, preprocessing, wavelength selection, model, RMSECV, RMSEP, R2, RPD, bias, notes).
5. **The locked test set is used once.** Record its sample IDs and hash in the Sample Log when it is created.
6. **Papers:** name PDFs `NN - Author Year - Title.pdf`, where `NN` is the paper's **Ref No.** in both tracking sheets (e.g. `28 - Lan 2022 - A Rapid Prediction Method of Moisture Content....pdf`). New papers take the next free number; tag topics in the matrices, not with folders.
7. **Files over 95 MB and videos** stay on PC + Google Drive only (they are kept off GitHub automatically).

## Adding another material later (coffee, gems, ...)

Create `04 Data/<material>/` (with the same `raw/`, `reference/`, `processed/` layout) and a matching results folder, e.g. `06 Results/Coffee Caffeine/`. The code in `05 Code/src/` stays shared.

## Plan (indicative)

| Phase | Dates | Output |
|-------|-------|--------|
| P1 Research and proposal | Sep - Oct 2026 | Proposal |
| P2 Ethics, protocol, pilot | Nov 2026 | SOPs, approval |
| P3 Moisture data collection | Dec 2026 - Feb 2027 | Dataset |
| P4 Cleaning and baseline | Jan - Feb 2027 | Locked test, baseline |
| P5 Multi-parameter synthesis | Jan - Mar 2027 | Feasibility matrix |
| P6 Preprocessing and selection | Feb - Mar 2027 | RQ2 results |
| P7 Models | Mar 2027 | Tuned models |
| P8 Evaluation and prototype | Mar - Apr 2027 | RQ1, RQ3 results, prototype |
| P9 Field feasibility | Apr 2027 | RQ4 results |
| P10 Write-up and viva | Apr - May 2027 | Thesis |
