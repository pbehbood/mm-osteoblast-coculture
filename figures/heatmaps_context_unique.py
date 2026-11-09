# -*- coding: utf-8 -*-
"""Regenerate the co-culture heatmaps on one shared red/blue scale,
using the published gene lists from Three_coculture_comparisons_DE.xlsx."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage, dendrogram
from config import COMBINED_MATRIX as MAT, THREE_CONTEXT_DE as XLS, OUT_DIR as OUT

BASE = MAT

# ---- one house style for every heatmap in the paper -------------------- #
CMAP = 'RdBu_r'            # blue = low, white = 0, red = high
VLIM = 2.0                 # symmetric row z-score limits
NORM = matplotlib.colors.Normalize(vmin=-VLIM, vmax=VLIM)   # symmetric, linear
CBAR_LABEL = 'Row $z$-score of log$_2$(FPKM + 1)'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                     'axes.linewidth': 0.6, 'figure.dpi': 300})

SAMPLES = [
    ('U266 MC1 TB12579', 'MC 1', 'U266 MC'), ('U266 MC2 TB12358', 'MC 2', 'U266 MC'),
    ('U266 MC3 TB12580', 'MC 3', 'U266 MC'),
    ('U266 CC1 TB12581', 'CC 1', 'U266 + hFOB'), ('U266 CC2 TB12361', 'CC 2', 'U266 + hFOB'),
    ('U266 CC3 TB12582', 'CC 3', 'U266 + hFOB'),
    ('MM1S MC1 TB12438', 'MC 1', 'MM1S MC (hFOB exp.)'), ('MM1S MC2 TB12439', 'MC 2', 'MM1S MC (hFOB exp.)'),
    ('MM1S MC3 TB12440', 'MC 3', 'MM1S MC (hFOB exp.)'),
    ('MM1S_hFOB CC1 TB12441', 'CC 1', 'MM1S + hFOB'), ('MM1S_hFOB CC2 TB12442', 'CC 2', 'MM1S + hFOB'),
    ('MM1S_hFOB CC3 TB12443', 'CC 3', 'MM1S + hFOB'),
    ('MM1S MC1 TB12986', 'MC 1', 'MM1S MC (HUVEC exp.)'), ('MM1S MC2 TB12987', 'MC 2', 'MM1S MC (HUVEC exp.)'),
    ('MM1S MC3 TB12988', 'MC 3', 'MM1S MC (HUVEC exp.)'),
    ('MM1S HUVEC CC1 TB12989', 'CC 1', 'MM1S + HUVEC'), ('MM1S HUVEC CC2 TB12990', 'CC 2', 'MM1S + HUVEC'),
    ('MM1S HUVEC CC3 TB12991', 'CC 3', 'MM1S + HUVEC'),
]
GROUP_COLOR = {
    'U266 MC': '#aecbe8', 'U266 + hFOB': '#1f6fb4',
    'MM1S MC (hFOB exp.)': '#fdbe6f', 'MM1S + hFOB': '#f57c20',
    'MM1S MC (HUVEC exp.)': '#b4df8c', 'MM1S + HUVEC': '#33a02c',
}
CLASS_COLOR = {'MM1S + hFOB-unique': '#f57c20', 'MM1S + HUVEC-unique': '#33a02c',
               'U266 + hFOB-unique': '#1f6fb4', 'Shared (MM1S, both partners)': '#7b4ea3',
               'Other DEG': '#b8b8b8'}
SHEET2CLASS = {'A-unique': 'MM1S + hFOB-unique', 'H-unique': 'MM1S + HUVEC-unique',
               'U-unique': 'U266 + hFOB-unique', 'A+H-shared': 'Shared (MM1S, both partners)'}


def load():
    df = pd.read_csv(MAT, sep='\t')
    df = df.rename(columns={df.columns[0]: 'gene'})
    df['gene'] = df['gene'].astype(str).str.strip()
    lists, allrows = {}, None
    for sheet, label in SHEET2CLASS.items():
        s = pd.read_excel(XLS, sheet_name=sheet)
        lists[label] = s[['gene_id', 'gene_name']].dropna()
    allrows = pd.read_excel(XLS, sheet_name='All DEGs any contrast')[['gene_id', 'gene_name']].dropna()
    return df, lists, allrows


def zmatrix(df, genes):
    """Rows = genes (by symbol), columns = the 18 samples, as row z-scores."""
    cols = [s[0] for s in SAMPLES]
    seen, want = set(), []
    for g in genes:                                   # de-duplicate, keep order
        if g not in seen:
            seen.add(g); want.append(g)
    sub = df[df['gene'].isin(want)].drop_duplicates('gene').set_index('gene')
    missing = [g for g in want if g not in sub.index]
    sub = sub.reindex([g for g in want if g in sub.index])
    m = np.log2(sub[cols].to_numpy(float) + 1)
    sd = m.std(1, ddof=0)
    keep = np.isfinite(m).all(1) & (sd > 0)          # drop flat / non-finite rows
    flat = list(sub.index[~keep])
    if missing or flat:
        print(f"      note: {len(missing)} not in matrix {missing}; "
              f"{len(flat)} with no variance across samples {flat}")
    m, sd = m[keep], sd[keep]
    return pd.DataFrame((m - m.mean(1, keepdims=True)) / sd[:, None],
                        index=sub.index[keep], columns=[s[1] for s in SAMPLES])


def draw(z, classes, title, fname, show_genes=True, figw=7.2):
    n = len(z)
    rowh = 0.135 if show_genes else max(0.012, 4.6 / max(n, 1))
    figh = max(3.2, min(11.0, n * rowh + 2.0))
    # reserve a fixed strip at the bottom for tick labels + legend, so short
    # figures don't push the legend up into the x labels
    BOT_IN, LEG_IN = 1.55, 0.80
    fig = plt.figure(figsize=(figw, figh + BOT_IN))
    figh_total = figh + BOT_IN
    gs = fig.add_gridspec(2, 3, width_ratios=[0.55, 0.055, 7], height_ratios=[0.42, 10],
                          wspace=0.03, hspace=0.03,
                          left=0.015, right=0.62 if show_genes else 0.84,
                          top=0.90, bottom=BOT_IN / figh_total)
    leg_y = LEG_IN / figh_total
    # cluster rows
    if n > 2:
        order = dendrogram(linkage(z.to_numpy(), 'average', 'euclidean'),
                           no_plot=True)['leaves']
    else:
        order = list(range(n))
    zz = z.iloc[order]
    cls = [classes[g] for g in zz.index] if classes else None

    ax_d = fig.add_subplot(gs[1, 0]); ax_d.axis('off')
    ax_c = fig.add_subplot(gs[1, 1])
    ax_h = fig.add_subplot(gs[1, 2])
    ax_g = fig.add_subplot(gs[0, 2])

    # sample group bar
    for i, (_, _, grp) in enumerate(SAMPLES):
        ax_g.add_patch(plt.Rectangle((i, 0), 1, 1, color=GROUP_COLOR[grp], lw=0))
    ax_g.set_xlim(0, len(SAMPLES)); ax_g.set_ylim(0, 1); ax_g.axis('off')

    # gene class bar
    if cls:
        for i, c in enumerate(cls):
            ax_c.add_patch(plt.Rectangle((0, i), 1, 1, color=CLASS_COLOR[c], lw=0))
        ax_c.set_xlim(0, 1); ax_c.set_ylim(len(cls), 0)
    ax_c.axis('off')

    im = ax_h.imshow(zz.to_numpy(), aspect='auto', cmap=CMAP, norm=NORM, interpolation='nearest')
    ax_h.set_xticks(range(len(SAMPLES)))
    ax_h.set_xticklabels([s[1] for s in SAMPLES], rotation=90, fontsize=6.5)
    if show_genes:
        ax_h.set_yticks(range(len(zz))); ax_h.set_yticklabels(zz.index, fontsize=6)
        ax_h.yaxis.tick_right()
    else:
        ax_h.set_yticks([])
    for sp in ax_h.spines.values():
        sp.set_visible(False)
    # white separators between the six groups
    for x in (2.5, 5.5, 8.5, 11.5, 14.5):
        ax_h.axvline(x, color='white', lw=1.6)

    cax = fig.add_axes([0.90 if show_genes else 0.885,
                        BOT_IN/figh_total + 0.42*(1-BOT_IN/figh_total), 0.015,
                        0.24*(1-BOT_IN/figh_total)])
    cb = fig.colorbar(im, cax=cax, ticks=[-VLIM, -1, 0, 1, VLIM], extend='both')
    cb.ax.set_yticklabels(['−2', '−1', '0', '1', '2'])
    cb.set_label(CBAR_LABEL, fontsize=7); cb.ax.tick_params(labelsize=6.5, length=2)
    cb.outline.set_linewidth(0.4)

    # legend below the plot so it can never collide with the gene labels
    handles = [Patch(facecolor=GROUP_COLOR[g], label=g) for g in
               ['U266 MC', 'U266 + hFOB', 'MM1S MC (hFOB exp.)', 'MM1S + hFOB',
                'MM1S MC (HUVEC exp.)', 'MM1S + HUVEC']]
    leg1 = fig.legend(handles=handles, title='Sample group', loc='upper left',
                      bbox_to_anchor=(0.015, leg_y), ncol=3, frameon=False, fontsize=6.3,
                      title_fontsize=6.8, handlelength=1.0, handleheight=1.0,
                      columnspacing=1.1, labelspacing=0.3)
    leg1._legend_box.align = 'left'
    if cls and len(set(cls)) > 1:
        h2 = [Patch(facecolor=CLASS_COLOR[c], label=c) for c in CLASS_COLOR if c in set(cls)]
        leg2 = fig.legend(handles=h2, title='Gene response pattern', loc='upper left',
                          bbox_to_anchor=(0.60, leg_y), ncol=1, frameon=False, fontsize=6.3,
                          title_fontsize=6.8, handlelength=1.0, handleheight=1.0,
                          labelspacing=0.3)
        leg2._legend_box.align = 'left'
        fig.add_artist(leg1)
    fig.suptitle(title, fontsize=9.5, x=0.40, y=0.965)
    fig.savefig(OUT + fname, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  wrote {fname}  ({len(zz)} genes)")


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    df, lists, allrows = load()
    classes = {}
    for label, t in lists.items():
        for g in t['gene_name']:
            classes[str(g).strip()] = label
    for g in allrows['gene_name']:
        classes.setdefault(str(g).strip(), 'Other DEG')
    print(f"class assignments: " +
          ", ".join(f"{k} {sum(1 for v in classes.values() if v==k)}" for k in CLASS_COLOR))

    # Figure: all DEGs across the three contexts (current Fig. 10)
    allg = [str(g).strip() for g in allrows['gene_name']]
    draw(zmatrix(df, allg), classes,
         'Differentially expressed genes across the three co-cultures',
         'heatmap_all_DEGs.png', show_genes=False, figw=7.4)

    # Context-unique panels (current Figs 12A, 13A, 14A)
    for label, fname, short in [
            ('MM1S + hFOB-unique', 'heatmap_MM1S_hFOB_unique.png', 'MM1S + hFOB'),
            ('MM1S + HUVEC-unique', 'heatmap_MM1S_HUVEC_unique.png', 'MM1S + HUVEC'),
            ('U266 + hFOB-unique', 'heatmap_U266_hFOB_unique.png', 'U266 + hFOB')]:
        g = [str(x).strip() for x in lists[label]['gene_name']]
        draw(zmatrix(df, g), classes,
             f'Genes specific to the {short} co-culture (n = {len(g)})', fname,
             show_genes=True)


if __name__ == '__main__':
    main()
