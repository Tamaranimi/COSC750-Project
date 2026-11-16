# COSC 750 — Stage 4 final submission

This branch is based on the original `final` tag, commit `916fe15836e10ceb8d0fb2c69d1e65cb49370ce6`. The historical tag is intentionally unchanged.

## Files

### Paper
- [Final paper (PDF)](Paper/COSC750_Stage4_Final_Paper.pdf)
- [Overleaf source and paper figures (ZIP)](Paper/COSC750_Stage4_Overleaf_Source.zip) — LaTeX, references, saved CSVs, figure-regeneration script, and Figures 2–4.

### Presentation
- [Corrected PowerPoint (PPTX)](Presentation/COSC750_Stage4_Final_Slides.pptx) — unchanged 17-slide deck.
- [Presentation PDF](Presentation/COSC750_Stage4_Final_Slides.pdf)

### Reproducibility
- [Slide chart reproduction package (ZIP)](Reproducibility/COSC750_Slide_Chart_Reproducibility.zip) — seven chart PNGs for slides 6–12, nine CSV inputs, Python script, and instructions.

## Chart regeneration

Extract the chart ZIP, then follow its included README. The script uses Python with NumPy and Matplotlib:

```bash
python -m pip install numpy matplotlib
python regenerate_slide_charts.py
```

The chart package reproduces the reported numerical content, not pixel-identical PowerPoint graphics or a complete rerun of the ML experiments. Some CSV values were transcribed from manuscript tables as documented in the chart-package README.

## Submission notes

- Paper page count is accepted.
- Presentation duration is **10 minutes total**.
- Existing `final` tag is preserved at `916fe15`.
- The presentation's embedded charts and design were not modified.
