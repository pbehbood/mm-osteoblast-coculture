"""Paths used by the analysis scripts.

Set DATA_DIR to the directory holding the expression matrices and the
gene-list spreadsheets; set OUT_DIR to where figures should be written.
The data files themselves are not distributed with this repository -- see
the Data availability statement in the manuscript.
"""
import os

DATA_DIR = os.environ.get('MM_DATA_DIR', 'data')
OUT_DIR = os.environ.get('MM_OUT_DIR', 'figures_out')

# expression matrices
COMBINED_MATRIX = os.path.join(DATA_DIR, 'U266 MM1S hFOB HUVEC combined data file.txt')
FOUR_CONDITION_MATRIX = os.path.join(DATA_DIR, 'MM1S_U266_RNA_data.txt')

# published gene lists
THREE_CONTEXT_DE = os.path.join(DATA_DIR, 'Three_coculture_comparisons_DE.xlsx')
MM1S_4GROUPS = os.path.join(DATA_DIR, 'heatmap_MM1S_4groups_genes.xlsx')
U266_MM1S_HFOB = os.path.join(DATA_DIR, 'heatmap_U266_MM1S_hFOB_genes.xlsx')

os.makedirs(OUT_DIR, exist_ok=True)
