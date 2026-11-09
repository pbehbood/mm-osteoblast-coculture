# -*- coding: utf-8 -*-
"""Remaining manuscript heatmaps (current Figs 6C, 7, 9, 21E) on the house style."""
import sys, os
import numpy as np, pandas as pd
from scipy import stats
from hmstyle import zmatrix, draw
from config import (COMBINED_MATRIX as COMBINED, FOUR_CONDITION_MATRIX as P7,
                    MM1S_4GROUPS, U266_MM1S_HFOB, OUT_DIR as OUT)


# ---- sample layouts --------------------------------------------------- #
S_FIG9 = [  # 12 MM.1S samples from the combined matrix
    ('MM1S MC1 TB12438', 'MC 1', 'MM1S MC (hFOB exp.)'), ('MM1S MC2 TB12439', 'MC 2', 'MM1S MC (hFOB exp.)'),
    ('MM1S MC3 TB12440', 'MC 3', 'MM1S MC (hFOB exp.)'),
    ('MM1S_hFOB CC1 TB12441', 'CC 1', 'MM1S + hFOB'), ('MM1S_hFOB CC2 TB12442', 'CC 2', 'MM1S + hFOB'),
    ('MM1S_hFOB CC3 TB12443', 'CC 3', 'MM1S + hFOB'),
    ('MM1S MC1 TB12986', 'MC 1', 'MM1S MC (HUVEC exp.)'), ('MM1S MC2 TB12987', 'MC 2', 'MM1S MC (HUVEC exp.)'),
    ('MM1S MC3 TB12988', 'MC 3', 'MM1S MC (HUVEC exp.)'),
    ('MM1S HUVEC CC1 TB12989', 'CC 1', 'MM1S + HUVEC'), ('MM1S HUVEC CC2 TB12990', 'CC 2', 'MM1S + HUVEC'),
    ('MM1S HUVEC CC3 TB12991', 'CC 3', 'MM1S + HUVEC'),
]
S_FIG7 = [  # hFOB / U266 / MM1S, monoculture vs co-culture (P7 matrix)
    ('hFOB MC1 TB12444', 'MC 1', 'hFOB MC'), ('hFOB MC2 TB12445', 'MC 2', 'hFOB MC'),
    ('hFOB MC3 TB12446', 'MC 3', 'hFOB MC'),
    ('hFOB_MM1S CC1 TB12447', 'CC 1', 'hFOB + MM1S'), ('hFOB_MM1S CC2 TB12448', 'CC 2', 'hFOB + MM1S'),
    ('hFOB_MM1S CC3 TB12449', 'CC 3', 'hFOB + MM1S'),
    ('U266 MC1 TB12579', 'MC 1', 'U266 MC'), ('U266 MC2 TB12358', 'MC 2', 'U266 MC'),
    ('U266 MC3 TB12580', 'MC 3', 'U266 MC'),
    ('U266 CC1 TB12581', 'CC 1', 'U266 + hFOB'), ('U266 CC2 TB12361', 'CC 2', 'U266 + hFOB'),
    ('U266 CC3 TB12582', 'CC 3', 'U266 + hFOB'),
    ('MM1S MC1 TB12438', 'MC 1', 'MM1S MC'), ('MM1S MC2 TB12439', 'MC 2', 'MM1S MC'),
    ('MM1S MC3 TB12440', 'MC 3', 'MM1S MC'),
    ('MM1S_hFOB CC1 TB12441', 'CC 1', 'MM1S + hFOB'), ('MM1S_hFOB CC2 TB12442', 'CC 2', 'MM1S + hFOB'),
    ('MM1S_hFOB CC3 TB12443', 'CC 3', 'MM1S + hFOB'),
]
S_15 = [  # the 4-condition MM.1S experiment plus its hFOB partners
    ('MM1S_hFOB CM1 TB12435', 'CM 1', 'MM1S + hFOB CM'), ('MM1S_hFOB CM2 TB12436', 'CM 2', 'MM1S + hFOB CM'),
    ('MM1S_hFOB CM3 TB12437', 'CM 3', 'MM1S + hFOB CM'),
    ('MM1S MC1 TB12438', 'MC 1', 'MM1S MC'), ('MM1S MC2 TB12439', 'MC 2', 'MM1S MC'),
    ('MM1S MC3 TB12440', 'MC 3', 'MM1S MC'),
    ('MM1S_hFOB CC1 TB12441', 'CC 1', 'MM1S + hFOB'), ('MM1S_hFOB CC2 TB12442', 'CC 2', 'MM1S + hFOB'),
    ('MM1S_hFOB CC3 TB12443', 'CC 3', 'MM1S + hFOB'),
    ('hFOB MC1 TB12444', 'MC 1', 'hFOB MC'), ('hFOB MC2 TB12445', 'MC 2', 'hFOB MC'),
    ('hFOB MC3 TB12446', 'MC 3', 'hFOB MC'),
    ('hFOB_MM1S CC1 TB12447', 'CC 1', 'hFOB + MM1S'), ('hFOB_MM1S CC2 TB12448', 'CC 2', 'hFOB + MM1S'),
    ('hFOB_MM1S CC3 TB12449', 'CC 3', 'hFOB + MM1S'),
]


def read(path):
    df = pd.read_csv(path, sep='\t').rename(columns=lambda c: c.strip())
    gc = 'gene_name' if 'gene_name' in df.columns else df.columns[0]
    df = df.rename(columns={gc: 'gene'})
    df['gene'] = df['gene'].astype(str).str.strip()
    return df


def welch(df, cc, mc):
    A = np.log2(df[cc].to_numpy(float) + 1); Bm = np.log2(df[mc].to_numpy(float) + 1)
    t, p = stats.ttest_ind(A, Bm, axis=1, equal_var=False)
    return A.mean(1) - Bm.mean(1), np.nan_to_num(p, nan=1.0)


def main():
    os.makedirs(OUT, exist_ok=True)
    comb, p7 = read(COMBINED), read(P7)

    # --- current Fig. 9: 151 genes, P<0.01 and FC>2.5 in either MM.1S contrast
    t = pd.read_excel(MM1S_4GROUPS)
    cls = {str(r.gene_name).strip(): r.passes_in for r in t.itertuples()}
    draw(zmatrix(comb, list(cls), S_FIG9), S_FIG9, cls,
         'Genes differentially expressed in the MM.1S co-cultures\n'
         '($P$ < 0.01 and fold change > 2.5 in at least one comparison)',
         OUT + '/heatmap_fig9_MM1S_hFOB_vs_HUVEC.png', show_genes=False, figw=6.2,
         class_title='Gene passes criteria in')

    # --- current Fig. 7: hFOB / U266 / MM.1S, monoculture vs co-culture
    t = pd.read_excel(U266_MM1S_HFOB)
    cls = {str(r.gene_name).strip(): r.passes_in for r in t.itertuples()}
    draw(zmatrix(p7, list(cls), S_FIG7), S_FIG7, cls,
         'Co-culture response across hFOB, U266 and MM.1S cells',
         OUT + '/heatmap_fig7_three_cell_types.png', show_genes=False, figw=6.8,
         class_title='Gene passes criteria in')

    # --- current Fig. 6C: MM.1S monoculture / conditioned medium / co-culture
    lfc, p = welch(p7, [s[0] for s in S_15[6:9]], [s[0] for s in S_15[3:6]])
    sel = (p < 0.01) & (np.abs(lfc) > np.log2(2.5))
    genes = list(pd.Series(p7['gene'][sel]).astype(str))
    print(f"  Fig 6C selection: {len(genes)} genes (MM.1S CC vs MC, P<0.01, FC>2.5)")
    draw(zmatrix(p7, genes, S_15), S_15, None,
         'MM.1S transcriptional response to osteoblast co-culture\n'
         'versus conditioned medium',
         OUT + '/heatmap_fig6C_MM1S_four_conditions.png', show_genes=False, figw=6.4)

    # --- current Fig. 21E: genes altered in hFOB cells by co-culture
    lfc, p = welch(p7, [s[0] for s in S_15[12:15]], [s[0] for s in S_15[9:12]])
    sel = (p < 0.05) & (np.abs(lfc) >= 1)
    genes = list(pd.Series(p7['gene'][sel]).astype(str))
    direction = {}
    for g, l in zip(p7['gene'][sel].astype(str), lfc[sel]):
        direction.setdefault(g, 'Up in co-culture' if l > 0 else 'Down in co-culture')
    print(f"  Fig 21E selection: {len(genes)} genes (hFOB CC vs MC, P<0.05, |log2FC|>=1)")
    draw(zmatrix(p7, genes, S_15), S_15, direction,
         'Genes altered in hFOB osteoblasts by co-culture with MM.1S cells',
         OUT + '/heatmap_fig21E_hFOB_response.png', show_genes=False, figw=6.4,
         class_title='Direction in co-culture')


if __name__ == '__main__':
    main()
