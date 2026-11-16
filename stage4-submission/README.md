# COSC 750 — Stage 4 Final Submission

The complete COSC 750 Stage 4 submission is maintained on the repository's `main` branch and identified by the `final` Git tag.

The original experimental verification commit (`916fe15836e10ceb8d0fb2c69d1e65cb49370ce6`) is preserved under the `stage4-core-verified` tag.

## Files

### Paper
- [Final paper (PDF)](Paper/COSC750_Stage4_Final_Paper.pdf)
- [Overleaf source and paper figures (ZIP)](Paper/COSC750_Stage4_Overleaf_Source.zip) — LaTeX, references, saved CSVs, figure-regeneration script, and Figures 2–4.

### Presentation
- [Corrected PowerPoint (PPTX)](Presentation/COSC750_Stage4_Final_Slides.pptx) — unchanged 17-slide deck.
- [Presentation PDF](Presentation/COSC750_Stage4_Final_Slides.pdf)

### Reproducibility
- [Slide chart reproduction package (ZIP)](Reproducibility/COSC750_Slide_Chart_Reproducibility.zip) — seven chart PNGs for slides 6–12, nine CSV inputs, Python script, and instructions.

## Chart Regeneration

Extract the chart ZIP and follow its included README. The script uses Python with NumPy and Matplotlib:

```bash
python -m pip install numpy matplotlib
python regenerate_slide_charts.py
```

The chart package reproduces the reported numerical content, not pixel-identical PowerPoint graphics or a complete rerun of the machine-learning experiments.

Some CSV values were transcribed from manuscript tables, as documented in the chart-package README.

## Experimental Reproducibility

The core matched-protocol experiments were verified twice using a fresh repository clone and the previously prepared CIC-IDS2017 Parquet dataset.

The verification reproduced the reported expected costs and six unseen attack-family recall values at the stated precision.

The original CSV-to-Parquet preparation was performed and repeated separately. The clean-clone verification did not independently recreate the prepared Parquet dataset from the original raw CSV files.

The additional Stage 4 ablation, calibration, and class-weight experiments are documented in the repository with saved results and reproduction commands. They were not independently rerun during the November 14 clean-clone verification.

See the repository's main `README.md` for the experimental protocol, verification scope, and reproduction instructions.

## Submission Notes

- Paper page count is accepted.
- Presentation duration is **10 minutes total**, as recorded in the submission notes.
- The `final` Git tag identifies the complete Stage 4 submission on `main`.
- The historical core verification commit is preserved under `stage4-core-verified`.
- The presentation's embedded charts and design were not modified.