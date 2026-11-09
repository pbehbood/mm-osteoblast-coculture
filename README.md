# Osteoblast–myeloma co-culture: analysis code

Analysis code for the study of reciprocal signalling between human osteoblasts
and multiple myeloma cells, in which osteoblast contact induces a type I
interferon programme in MM.1S cells and myeloma contact drives contact-dependent
CXCL1/CXCL8 secretion by osteoblasts.

## Contents

```
figures/
  config.py                      paths to the input matrices and gene lists
  hmstyle.py                     shared heatmap style (RdBu_r, row z-score, ±2)
  three_context_de.py            differential expression across the three
                                 co-culture contexts, plus the
                                 difference-of-differences test used to call
                                 context-specific genes
  heatmaps_context_unique.py     all-DEG and context-unique heatmaps
  heatmaps_additional.py         remaining expression heatmaps
survival/
  lasso_interferon_signature.py  LASSO-penalised Cox model over the interferon
                                 gene signature and Kaplan–Meier stratification
```

## Method summary

**Differential expression.** Three co-culture contexts (U266 + hFOB, MM.1S +
hFOB, MM.1S + HUVEC) were each compared with their own matched monoculture
control, n = 3 biological replicates per group. Genes with a maximum FPKM ≤ 1
across the six samples of a contrast were excluded, values were
log2(x + 1)-transformed, and differential expression was assessed by Welch's
t-test with Benjamini–Hochberg correction (padj < 0.05, |log2FC| ≥ 1).

**Context specificity.** Differences between contexts were tested with a
Welch–Satterthwaite statistic on the difference of the two effect sizes, with
FDR correction within each comparison family. A gene was called
context-specific if it was differentially expressed in that context, differed
significantly from both other contexts, and had the largest |log2FC| of the
three.

**Heatmaps.** All expression heatmaps use one scale: row z-scores of
log2(FPKM + 1) on a symmetric ±2 `RdBu_r` map with a shared colour bar, rows
clustered by average linkage on Euclidean distance.

**Survival.** A LASSO-penalised Cox model (`CoxnetSurvivalAnalysis`,
`l1_ratio = 1.0`, `alpha = 0.1`) was fitted to standardised expression of 19
interferon-stimulated genes; patients were split at the median risk score and
compared by log-rank test. The penalty was fixed rather than cross-validated,
and the model was fitted and evaluated in the same cohort.

## Data

The expression matrices, methylation data and clinical annotations are not
distributed here. See the Data availability statement in the manuscript for
repository accessions. Set `MM_DATA_DIR` and `MM_OUT_DIR`, or edit
`figures/config.py`, to point at local copies.

## Running

```bash
pip install -r requirements.txt
export MM_DATA_DIR=/path/to/data MM_OUT_DIR=/path/to/output
python figures/three_context_de.py
python figures/heatmaps_context_unique.py
python figures/heatmaps_additional.py
python survival/lasso_interferon_signature.py
```

## Citation

If you use this code, please cite the accompanying manuscript. Citation details
will be added here on publication.

## Licence

MIT — see [LICENSE](LICENSE).
