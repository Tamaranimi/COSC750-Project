# COSC 750 — Stage 4 submission synchronization

This branch was created from the **existing immutable `final` tag**, commit `916fe15836e10ceb8d0fb2c69d1e65cb49370ce6`.

## Intended submission files

- `Paper/COSC750_Stage4_Final_Paper.pdf` — corrected final paper
- `Paper/COSC750_Stage4_Overleaf_Source.zip` — paper source, CSVs, regeneration script, Figures 2–4
- `Presentation/COSC750_Stage4_Final_Slides.pptx` — corrected 17-slide PowerPoint, **unchanged**
- `Presentation/COSC750_Stage4_Final_Slides.pdf` — corresponding presentation PDF
- `Reproducibility/COSC750_Slide_Chart_Reproducibility.zip` — seven chart PNGs, nine CSV files, `regenerate_slide_charts.py`, README

## Reproduction notes

The slide-chart package uses `numpy` and `matplotlib`:

```bash
python -m pip install numpy matplotlib
python regenerate_slide_charts.py
```

The charts represent slides 6–12. Source tables were transcribed from the manuscript where indicated in the chart-package README. Regenerated charts are numerically equivalent but not pixel-identical to the PowerPoint charts; this is **not** a rerun of the full machine-learning experiments.

## Verification and submission constraints

- Existing `final` tag remains untouched at `916fe15`.
- The paper's page count is accepted.
- Presentation time is **10 minutes total**.
- Do not replace or restyle the corrected PowerPoint's embedded charts.
- Do not claim the five binary deliverables have been synchronized until each has been uploaded and checked on this branch.

**Status:** This manifest was committed first. Binary artifacts require separate transfer and verification.
