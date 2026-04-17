import streamlit as st
import pandas as pd
import numpy as np
import os

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.Chem.Scaffolds import MurckoScaffold

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.stats import mannwhitneyu

import umap
from collections import Counter

# =========================
# CONFIG
# =========================
MAX_MOLECULES = 1000

st.title("Structural Diversity Analysis")

# =========================
# DATA LOADING
# =========================
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("No file uploaded. Using example_dataset.csv from local directory.")
    if os.path.exists("example_dataset.csv"):
        df = pd.read_csv("example_dataset.csv")
    else:
        st.error("No example_dataset.csv found in directory.")
        st.stop()

# =========================
# CHECK COLUMNS
# =========================
if not {"SMILES_STANDARDIZED", "CLASS"}.issubset(df.columns):
    st.error("CSV must contain 'SMILES_STANDARDIZED' and 'CLASS'")
    st.stop()

df = df[['SMILES_STANDARDIZED', 'CLASS']].dropna()

if len(df) > MAX_MOLECULES:
    st.warning(f"Dataset too large ({len(df)}). Subsampling to {MAX_MOLECULES}")
    df = df.sample(MAX_MOLECULES, random_state=42)

# =========================
# FUNCTIONS
# =========================
def compute_fps(smiles, radius, nbits):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    fps, valid_idx = [], []

    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fps.append(gen.GetFingerprint(mol))
            valid_idx.append(i)

    return fps, valid_idx


def tanimoto_matrix(fps):
    n = len(fps)
    sim = np.zeros((n, n))
    for i in range(n):
        sim[i, :] = BulkTanimotoSimilarity(fps[i], fps)
    return sim


def flatten_upper(mat):
    return mat[np.triu_indices_from(mat, k=1)]


def scaffold_metrics(smiles):
    scaffolds = []

    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            scaf = MurckoScaffold.GetScaffoldForMol(mol)
            scaffolds.append(Chem.MolToSmiles(scaf))

    counts = Counter(scaffolds)
    total = len(scaffolds)

    return counts, total


# =========================
# PARAMETERS
# =========================
st.sidebar.header("Fingerprint parameters")
radius = st.sidebar.slider("Radius", 1, 4, 2)
nbits = st.sidebar.selectbox("FP length", [512, 1024, 2048, 4096], 2)

run = st.button("Run analysis")

# =========================
# RUN
# =========================
if run:

    smiles = df.SMILES_STANDARDIZED.tolist()
    classes = df.CLASS.values

    # =====================
    # FINGERPRINTS
    # =====================
    fps, valid_idx = compute_fps(smiles, radius, nbits)

    df = df.iloc[valid_idx].reset_index(drop=True)
    classes = df.CLASS.values
    smiles = df.SMILES_STANDARDIZED.tolist()

    # =====================
    # SIMILARITY
    # =====================
    sim = tanimoto_matrix(fps)

    idx1 = np.where(classes == 1)[0]
    idx0 = np.where(classes == 0)[0]

    sim11 = sim[np.ix_(idx1, idx1)]
    sim00 = sim[np.ix_(idx0, idx0)]
    sim10 = sim[np.ix_(idx1, idx0)]

    v11 = flatten_upper(sim11)
    v00 = flatten_upper(sim00)
    v10 = sim10.flatten()

    # =========================
    # STATISTICAL TEST
    # =========================
    st.subheader("Statistical test (Mann–Whitney U)")

    st.markdown("""
We compare whether **within-class similarity (Class 1–1)** is significantly higher than
**between-class similarity (Class 1–0)** using a non-parametric Mann–Whitney test.

This tests whether active compounds are structurally more similar to each other than to inactive ones.
""")

    stat, pval = mannwhitneyu(v11, v10, alternative="greater")

    st.write(f"p-value: {pval:.3e}")

    # =========================
    # HEATMAP
    # =========================
    st.subheader("Clustered Heatmap")

    st.markdown("""
This heatmap shows **pairwise Tanimoto similarity between all molecules**, reordered by hierarchical clustering.

How to read it:
- Blocks of high intensity indicate structurally similar clusters
- Row/column colors indicate class membership
- If classes separate well, blocks will align with colors
""")

    dist = 1 - sim
    link = linkage(squareform(dist, checks=False), method="average")

    colors = {0: "#1f77b4", 1: "#d62728"}
    row_colors = pd.Series(classes).map(colors).to_numpy()

    fig = sns.clustermap(
        sim,
        row_linkage=link,
        col_linkage=link,
        row_colors=row_colors,
        col_colors=row_colors,
        cmap="viridis",
        xticklabels=False,
        yticklabels=False,
        figsize=(8, 8)
    )

    st.pyplot(fig)

    # =========================
    # DISTRIBUTIONS
    # =========================
    st.subheader("Similarity distributions")

    st.markdown("""
These distributions show how similarity values are distributed across:
- Class 1 vs Class 1 (intra-active)
- Class 0 vs Class 0 (intra-inactive)
- Class 1 vs Class 0 (inter-class)

Separation between curves suggests structural differentiation between classes.
""")

    fig2, ax = plt.subplots()

    sns.kdeplot(v11, label="Class 1-1", fill=True, ax=ax)
    sns.kdeplot(v00, label="Class 0-0", fill=True, ax=ax)
    sns.kdeplot(v10, label="Class 1-0", fill=True, ax=ax)

    ax.legend()
    st.pyplot(fig2)

    # =========================
    # UMAP
    # =========================
    st.subheader("UMAP projection")

    st.markdown("""
UMAP reduces high-dimensional fingerprint space into 2D.

Each point represents a molecule.
Proximity reflects structural similarity.
""")

    fp_array = np.array([list(fp) for fp in fps])
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="jaccard")
    emb = reducer.fit_transform(fp_array)

    fig3, ax = plt.subplots()
    ax.scatter(emb[:, 0], emb[:, 1], c=classes, cmap="coolwarm", s=20)
    st.pyplot(fig3)

    # =========================
    # SCAFFOLDS
    # =========================
    st.subheader("Scaffold diversity (Murcko scaffolds)")

    st.markdown("""
Scaffolds represent the **core molecular framework** of compounds.

This analysis evaluates:
- Structural diversity at the core level
- Dominant chemical motifs in each class
""")

    counts_all, total_all = scaffold_metrics(smiles)
    counts_1, total_1 = scaffold_metrics(df[df.CLASS == 1]['SMILES_STANDARDIZED'])
    counts_0, total_0 = scaffold_metrics(df[df.CLASS == 0]['SMILES_STANDARDIZED'])

    def top_table(counts, name):
        top5 = counts.most_common(5)
        return pd.DataFrame(top5, columns=[f"{name}_scaffold", "count"])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("All")
        st.dataframe(top_table(counts_all, "All"))

    with col2:
        st.write("Class 1")
        st.dataframe(top_table(counts_1, "Class1"))

    with col3:
        st.write("Class 0")
        st.dataframe(top_table(counts_0, "Class0"))

    # =========================
    # SUMMARY MESSAGE
    # =========================
    st.subheader("Key interpretation")

    delta = np.mean(v11) - np.mean(v10)

    if delta < 0.02:
        st.warning("Very weak structural separation between classes")
    elif delta < 0.05:
        st.info("Moderate structural separation")
    else:
        st.success("Strong structural enrichment detected")
