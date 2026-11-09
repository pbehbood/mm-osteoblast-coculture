# -*- coding: utf-8 -*-
"""Shared house style for every expression heatmap in the manuscript."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage, dendrogram

CMAP = 'RdBu_r'
VLIM = 2.0
NORM = matplotlib.colors.Normalize(vmin=-VLIM, vmax=VLIM)
CBAR_LABEL = 'Row $z$-score of log$_2$(FPKM + 1)'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                     'axes.linewidth': 0.6, 'figure.dpi': 300})

# one colour per biological group, pale = monoculture, saturated = co-culture
GROUP_COLOR = {
    'U266 MC': '#aecbe8', 'U266 + hFOB': '#1f6fb4',
    'MM1S MC': '#fdbe6f', 'MM1S MC (hFOB exp.)': '#fdbe6f', 'MM1S + hFOB': '#f57c20',
    'MM1S MC (HUVEC exp.)': '#b4df8c', 'MM1S + HUVEC': '#33a02c',
    'MM1S + hFOB CM': '#d9d2e9',
    'hFOB MC': '#d9d9d9', 'hFOB + MM1S': '#6e6e6e',
}
CLASS_COLOR = {
    'MM1S + hFOB only': '#f57c20', 'MM1S + HUVEC only': '#33a02c',
    'U266 + hFOB only': '#1f6fb4', 'Both comparisons': '#7b4ea3',
    'MM1S + hFOB-unique': '#f57c20', 'MM1S + HUVEC-unique': '#33a02c',
    'U266 + hFOB-unique': '#1f6fb4', 'Shared (MM1S, both partners)': '#7b4ea3',
    'Other DEG': '#b8b8b8',
    'Up in co-culture': '#c0392b', 'Down in co-culture': '#2471a3',
}


def zmatrix(df, genes, samples, gene_col='gene'):
    """samples: list of (column, tick label, group). Returns row z-scores."""
    cols = [s[0] for s in samples]
    seen, want = set(), []
    for g in genes:
        if g not in seen:
            seen.add(g); want.append(g)
    sub = df[df[gene_col].isin(want)].drop_duplicates(gene_col).set_index(gene_col)
    missing = [g for g in want if g not in sub.index]
    sub = sub.reindex([g for g in want if g in sub.index])
    m = np.log2(sub[cols].to_numpy(float) + 1)
    sd = m.std(1, ddof=0)
    keep = np.isfinite(m).all(1) & (sd > 0)
    flat = list(sub.index[~keep])
    if missing or flat:
        print(f"      note: {len(missing)} absent {missing[:6]}; {len(flat)} flat {flat[:6]}")
    m, sd = m[keep], sd[keep]
    return pd.DataFrame((m - m.mean(1, keepdims=True)) / sd[:, None],
                        index=sub.index[keep], columns=[s[1] for s in samples])


def draw(z, samples, classes, title, path, show_genes=True, figw=7.2,
         class_title='Gene response pattern'):
    n = len(z)
    rowh = 0.135 if show_genes else max(0.012, 4.6 / max(n, 1))
    figh = max(3.2, min(11.0, n * rowh + 2.0))
    BOT_IN, LEG_IN = 1.55, 0.80
    fig = plt.figure(figsize=(figw, figh + BOT_IN))
    tot = figh + BOT_IN
    gs = fig.add_gridspec(2, 3, width_ratios=[0.55, 0.055, 7], height_ratios=[0.42, 10],
                          wspace=0.03, hspace=0.03, left=0.015,
                          right=0.62 if show_genes else 0.84,
                          top=0.90, bottom=BOT_IN / tot)
    leg_y = LEG_IN / tot

    order = dendrogram(linkage(z.to_numpy(), 'average', 'euclidean'),
                       no_plot=True)['leaves'] if n > 2 else list(range(n))
    zz = z.iloc[order]
    cls = [classes[g] for g in zz.index] if classes else None

    fig.add_subplot(gs[1, 0]).axis('off')
    ax_c = fig.add_subplot(gs[1, 1]); ax_h = fig.add_subplot(gs[1, 2])
    ax_g = fig.add_subplot(gs[0, 2])

    for i, (_, _, grp) in enumerate(samples):
        ax_g.add_patch(plt.Rectangle((i, 0), 1, 1, color=GROUP_COLOR[grp], lw=0))
    ax_g.set_xlim(0, len(samples)); ax_g.set_ylim(0, 1); ax_g.axis('off')

    if cls:
        for i, c in enumerate(cls):
            ax_c.add_patch(plt.Rectangle((0, i), 1, 1, color=CLASS_COLOR[c], lw=0))
        ax_c.set_xlim(0, 1); ax_c.set_ylim(len(cls), 0)
    ax_c.axis('off')

    im = ax_h.imshow(zz.to_numpy(), aspect='auto', cmap=CMAP, norm=NORM,
                     interpolation='nearest')
    ax_h.set_xticks(range(len(samples)))
    ax_h.set_xticklabels([s[1] for s in samples], rotation=90, fontsize=6.5)
    if show_genes:
        ax_h.set_yticks(range(len(zz))); ax_h.set_yticklabels(zz.index, fontsize=6)
        ax_h.yaxis.tick_right()
    else:
        ax_h.set_yticks([])
    for sp in ax_h.spines.values():
        sp.set_visible(False)
    # white rule wherever the group changes
    groups = [s[2] for s in samples]
    for i in range(1, len(groups)):
        if groups[i] != groups[i - 1]:
            ax_h.axvline(i - 0.5, color='white', lw=1.6)

    cax = fig.add_axes([0.90 if show_genes else 0.885,
                        BOT_IN / tot + 0.42 * (1 - BOT_IN / tot), 0.015,
                        0.24 * (1 - BOT_IN / tot)])
    cb = fig.colorbar(im, cax=cax, ticks=[-VLIM, -1, 0, 1, VLIM], extend='both')
    cb.ax.set_yticklabels(['−2', '−1', '0', '1', '2'])
    cb.set_label(CBAR_LABEL, fontsize=7); cb.ax.tick_params(labelsize=6.5, length=2)
    cb.outline.set_linewidth(0.4)

    seen, ordered = set(), []
    for _, _, g in samples:
        if g not in seen:
            seen.add(g); ordered.append(g)
    h1 = [Patch(facecolor=GROUP_COLOR[g], label=g) for g in ordered]
    leg1 = fig.legend(handles=h1, title='Sample group', loc='upper left',
                      bbox_to_anchor=(0.015, leg_y), ncol=3, frameon=False,
                      fontsize=6.3, title_fontsize=6.8, handlelength=1.0,
                      handleheight=1.0, columnspacing=1.1, labelspacing=0.3)
    leg1._legend_box.align = 'left'
    if cls and len(set(cls)) > 1:
        h2 = [Patch(facecolor=CLASS_COLOR[c], label=c)
              for c in CLASS_COLOR if c in set(cls)]
        leg2 = fig.legend(handles=h2, title=class_title, loc='upper left',
                          bbox_to_anchor=(0.60, leg_y), frameon=False, fontsize=6.3,
                          title_fontsize=6.8, handlelength=1.0, handleheight=1.0,
                          labelspacing=0.3)
        leg2._legend_box.align = 'left'
        fig.add_artist(leg1)
    fig.suptitle(title, fontsize=9.5, x=0.40, y=0.965)
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  wrote {path.split('/')[-1]}  ({len(zz)} genes)")
