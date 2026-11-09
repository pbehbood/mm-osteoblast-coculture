# -*- coding: utf-8 -*-
"""Reproduce the three-context co-culture analysis from the manuscript Methods.
Rows are kept as-is (no collapsing of duplicate symbols), which reproduces the
published retained-gene counts exactly."""
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from config import COMBINED_MATRIX as MAT, OUT_DIR as OUT


GROUPS = {
    'U266_MC':      ['U266 MC1 TB12579', 'U266 MC2 TB12358', 'U266 MC3 TB12580'],
    'U266_CC':      ['U266 CC1 TB12581', 'U266 CC2 TB12361', 'U266 CC3 TB12582'],
    'MM1S_MC_hFOB': ['MM1S MC1 TB12438', 'MM1S MC2 TB12439', 'MM1S MC3 TB12440'],
    'MM1S_hFOB_CC': ['MM1S_hFOB CC1 TB12441', 'MM1S_hFOB CC2 TB12442', 'MM1S_hFOB CC3 TB12443'],
    'MM1S_MC_HUV':  ['MM1S MC1 TB12986', 'MM1S MC2 TB12987', 'MM1S MC3 TB12988'],
    'MM1S_HUV_CC':  ['MM1S HUVEC CC1 TB12989', 'MM1S HUVEC CC2 TB12990', 'MM1S HUVEC CC3 TB12991'],
}
CONTRASTS = {
    'U266_hFOB':  ('U266_CC', 'U266_MC'),
    'MM1S_hFOB':  ('MM1S_hFOB_CC', 'MM1S_MC_hFOB'),
    'MM1S_HUVEC': ('MM1S_HUV_CC', 'MM1S_MC_HUV'),
}
DATE_FIX = {}
for n in range(1, 15):
    DATE_FIX[f'{n}-Sep'] = f'SEPT{n}'; DATE_FIX[f'{n}-Mar'] = f'MARCH{n}'
    DATE_FIX[f'Sep-{n:02d}'] = f'SEPT{n}'; DATE_FIX[f'Mar-{n:02d}'] = f'MARCH{n}'


def load():
    df = pd.read_csv(MAT, sep='\t').rename(columns=lambda c: c.strip())
    df = df.rename(columns={df.columns[0]: 'gene'})
    df['gene'] = df['gene'].astype(str).str.strip()
    nfix = df['gene'].isin(DATE_FIX).sum()
    df['gene'] = df['gene'].replace(DATE_FIX)
    print(f"loaded {len(df)} rows; repaired {nfix} date-mangled symbols")
    return df


def contrast(df, cc, mc):
    """Welch t-test on log2(FPKM+1); returns per-row stats plus Welch df."""
    A = df[GROUPS[cc]].to_numpy(float); B = df[GROUPS[mc]].to_numpy(float)
    keep = np.maximum(A.max(1), B.max(1)) > 1
    A, B = np.log2(A[keep] + 1), np.log2(B[keep] + 1)
    na, nb = A.shape[1], B.shape[1]
    va, vb = A.var(1, ddof=1) / na, B.var(1, ddof=1) / nb
    se2 = va + vb
    lfc = A.mean(1) - B.mean(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = lfc / np.sqrt(se2)
        dfree = se2 ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1))
    t = np.nan_to_num(t); dfree = np.nan_to_num(dfree, nan=1.0)
    p = np.nan_to_num(2 * stats.t.sf(np.abs(t), np.maximum(dfree, 1)), nan=1.0)
    padj = multipletests(p, method='fdr_bh')[1]
    idx = np.flatnonzero(keep)
    return pd.DataFrame({'gene': df['gene'].to_numpy()[idx], 'lfc': lfc, 'p': p,
                         'padj': padj, 'se2': se2, 'df': dfree}, index=idx)


def main():
    df = load()
    res = {n: contrast(df, cc, mc) for n, (cc, mc) in CONTRASTS.items()}
    for n, r in res.items():
        deg = ((r.padj < 0.05) & (r.lfc.abs() >= 1)).sum()
        print(f"  {n:11} retained {len(r):6d}  DEGs {deg:5d}")
    print("  published retained: 15758 / 14223 / 13641\n")

    names = list(CONTRASTS)
    diff = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            A, B = names[i], names[j]
            idx = res[A].index.intersection(res[B].index)
            a, b = res[A].loc[idx], res[B].loc[idx]
            se2 = a.se2 + b.se2
            with np.errstate(divide='ignore', invalid='ignore'):
                t = (a.lfc - b.lfc) / np.sqrt(se2)
                # Welch-Satterthwaite df for the difference of two effects
                dfree = se2 ** 2 / (a.se2 ** 2 / a['df'].clip(lower=1)
                                    + b.se2 ** 2 / b['df'].clip(lower=1))
            t = t.fillna(0); dfree = dfree.fillna(1).clip(lower=1)
            p = pd.Series(np.nan_to_num(2 * stats.t.sf(np.abs(t), dfree), nan=1.0), index=idx)
            diff[(A, B)] = pd.Series(multipletests(p.to_numpy(), method='fdr_bh')[1], index=idx)
            print(f"  diff {A} vs {B}: {(diff[(A,B)] < 0.05).sum()} rows differ")

    def differs(g, A, B):
        s = diff.get((A, B)) if (A, B) in diff else diff[(B, A)]
        return g in s.index and s[g] < 0.05

    rows = sorted(set().union(*[set(r.index) for r in res.values()]))
    cls = {}
    for g in rows:
        isdeg, lfcs = {}, {}
        for n in names:
            r = res[n]
            if g in r.index:
                lfcs[n] = r.lfc[g]
                isdeg[n] = (r.padj[g] < 0.05) and abs(r.lfc[g]) >= 1
            else:
                lfcs[n] = 0.0; isdeg[n] = False
        if not any(isdeg.values()):
            continue
        biggest = max(names, key=lambda n: abs(lfcs[n]))
        for n in names:
            if isdeg[n] and biggest == n and all(differs(g, n, m) for m in names if m != n):
                cls[g] = n + '_unique'; break
        else:
            degn = [n for n in names if isdeg[n]]
            if (len(degn) == 2 and 'U266_hFOB' not in degn
                    and np.sign(lfcs[degn[0]]) == np.sign(lfcs[degn[1]])
                    and not differs(g, *degn)):
                cls[g] = 'shared_MM1S'
            else:
                cls[g] = 'other_DEG'

    s = pd.Series(cls, name='class')
    print("\nclasses:")
    for k, v in s.value_counts().items():
        print(f"  {k:22} {v}")
    print("  published: hFOB-unique 71, HUVEC-unique 30, U266-unique 10, shared 9, other 195")

    out = pd.DataFrame({'gene': df['gene'].to_numpy()[s.index], 'class': s.to_numpy()}, index=s.index)
    out.to_csv(OUT + 'gene_classes.csv')
    for n, r in res.items():
        r.to_csv(OUT + f'contrast_{n}.csv')
    df.to_pickle(OUT + 'fpkm.pkl')
    print("\nwrote gene_classes.csv, contrast_*.csv, fpkm.pkl")


if __name__ == '__main__':
    main()
