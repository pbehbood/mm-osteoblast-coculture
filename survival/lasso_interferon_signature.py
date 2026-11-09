"""
LASSO-penalised Cox model over an interferon-stimulated gene signature,
with Kaplan-Meier stratification by the resulting risk score.

This reproduces the survival analysis reported for the public myeloma
expression series (GEO GSE24080). It is a cleaned, scripted form of the
original exploratory notebook; the statistical procedure is unchanged.

Note on the penalty: a single fixed penalty (alpha = 0.1, l1_ratio = 1.0,
i.e. pure LASSO) is used rather than a cross-validated value, and the model
is fitted and evaluated on the same cohort. The resulting log-rank p-value
is therefore descriptive of this series and not an out-of-sample estimate.

Inputs expected in DATA_DIR:
  expression.txt   genes x samples, tab-separated, first column = gene symbol
  clinical.txt     one row per sample, with survival time and event columns

Not distributed here: see the Data availability statement in the manuscript.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

DATA_DIR = os.environ.get('MM_DATA_DIR', 'data')
OUT_DIR = os.environ.get('MM_OUT_DIR', 'figures_out')
os.makedirs(OUT_DIR, exist_ok=True)

# Interferon-stimulated genes up-regulated in MM.1S + hFOB co-culture
IFN_GENES = ['MX1', 'MX2', 'IFIT1', 'IFIT3', 'IFI6', 'IFI16', 'IRF3', 'IRF7',
             'IRF9', 'ISG15', 'STAT1', 'IFITM1', 'IFITM3', 'OAS2', 'OAS3',
             'OASL', 'RSAD2', 'GBP2', 'XAF1']

TIME_COL, EVENT_COL = 'OS_months', 'OS_event'
ALPHA, L1_RATIO, MAX_ITER = 0.1, 1.0, 10000


def load():
    expr = pd.read_csv(os.path.join(DATA_DIR, 'expression.txt'), sep='\t', index_col=0)
    clin = pd.read_csv(os.path.join(DATA_DIR, 'clinical.txt'), sep='\t', index_col=0)
    shared = expr.columns.intersection(clin.index)
    return expr[shared], clin.loc[shared]


def fit_signature(expr, clin):
    present = [g for g in IFN_GENES if g in expr.index]
    missing = [g for g in IFN_GENES if g not in expr.index]
    if missing:
        print(f'  not on the array: {missing}')
    subset = expr.loc[present]                       # genes x samples

    # samples x genes, standardised so coefficients are comparable
    X = StandardScaler().fit_transform(subset.T.to_numpy(float))
    y = Surv.from_arrays(event=clin[EVENT_COL].astype(bool).to_numpy(),
                         time=clin[TIME_COL].astype(float).to_numpy())

    model = CoxnetSurvivalAnalysis(l1_ratio=L1_RATIO, alphas=[ALPHA], max_iter=MAX_ITER)
    model.fit(X, y)
    coef = model.coef_[:, 0]

    weights = pd.Series(coef, index=present).sort_values()
    retained = weights[weights != 0]
    print(f'  {len(retained)}/{len(present)} genes retained at alpha={ALPHA}')
    print(retained.to_string())

    score = pd.Series(X.dot(coef), index=subset.columns, name='LASSO_score')
    return score, weights


def stratify_and_plot(score, clin, path):
    d = clin.copy()
    d['score'] = score
    d['high'] = d['score'] >= d['score'].median()    # median split

    a, b = d[d.high], d[~d.high]
    res = logrank_test(a[TIME_COL], b[TIME_COL],
                       event_observed_A=a[EVENT_COL], event_observed_B=b[EVENT_COL])

    cph = CoxPHFitter().fit(d[[TIME_COL, EVENT_COL, 'high']].astype(float),
                            duration_col=TIME_COL, event_col=EVENT_COL)
    hr = float(np.exp(cph.params_['high']))
    lo, hi = np.exp(cph.confidence_intervals_.loc['high'].to_numpy())
    print(f'  log-rank p = {res.p_value:.3g};  HR (high vs low) = {hr:.2f} '
          f'({lo:.2f}-{hi:.2f})')

    fig, ax = plt.subplots(figsize=(5, 4))
    for mask, label, colour in [(d.high, 'High', '#d1603d'), (~d.high, 'Low', '#4878a8')]:
        km = KaplanMeierFitter()
        km.fit(d.loc[mask, TIME_COL], d.loc[mask, EVENT_COL], label=label)
        km.plot_survival_function(ax=ax, ci_show=True, color=colour)
    ax.set_xlabel('Time (months)'); ax.set_ylabel('Survival probability')
    ax.set_title('Survival by LASSO interferon risk score')
    txt = 'p < 0.001' if res.p_value < 0.001 else f'p = {res.p_value:.3f}'
    ax.text(0.05, 0.12, txt, transform=ax.transAxes)
    ax.legend(title='Risk group', frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)
    print(f'  wrote {path}')
    return res.p_value, hr


def main():
    expr, clin = load()
    print(f'loaded {expr.shape[0]} genes x {expr.shape[1]} samples with outcome data')
    score, weights = fit_signature(expr, clin)
    weights.to_csv(os.path.join(OUT_DIR, 'lasso_coefficients.csv'))
    stratify_and_plot(score, clin, os.path.join(OUT_DIR, 'km_lasso_ifn_score.png'))


if __name__ == '__main__':
    main()
