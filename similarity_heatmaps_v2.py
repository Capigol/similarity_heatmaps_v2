import streamlit as st
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.Chem.Scaffolds import MurckoScaffold

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.stats import mannwhitneyu

from sklearn.manifold import TSNE
import umap

from collections import Counter

# =========================
# CONFIG
# =========================
MAX_MOLECULES = 1000

# =========================
# UI
# =========================
st.title("Structural Diversity Analysis")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

st.sidebar.header("Fingerprint parameters")
radius = st.sidebar.slider("Radius", 1, 4, 2)
nbits = st.sidebar.selectbox("FP length", [512, 1024, 2048, 4096], 2)

run = st.button("Run analysis")

# =========================
# FUNCTIONS
# =========================
def compute_fps(smiles, radius, nbits):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    fps, valid_idx, invalid = [], [], 0

    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fps.append(gen.GetFingerprint(mol))
            valid_idx.append(i)
        else:
            invalid += 1

    return fps, valid_idx, invalid


def tanimoto_matrix(fps):
    n = len(fps)
    sim = np.zeros((n, n))
    for i in range(n):
        sim[i, :] = BulkTanimotoSimilarity(fps[i], fps)
    return sim


def flatten_upper(mat):
    return mat[np.triu_indices_from(mat, k=1)]


def nearest_neighbor(sim):
    sim = sim.copy()
    np.fill_diagonal(sim, 0)
    return sim.max(axis=1)


def scaffold_metrics(smiles):
    scaffolds = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            scaf = MurckoScaffold.GetScaffoldForMol(mol)
            scaffolds.append(Chem.MolToSmiles(scaf))

    counts = Counter(scaffolds)
    total = len(scaffolds)
    probs = np.array(list(counts.values())) / total

    return {
        "n_scaffolds": len(counts),
        "ratio": len(counts)/total,
        "entropy": -np.sum(probs * np.log(probs))
    }

# =========================
# MAIN
# =========================
if uploaded_file and run:

    df = pd.read_csv(uploaded_file)

    if not {"SMILES_STANDARDIZED", "CLASS"}.issubset(df.columns):
        st.error("Missing required columns")
        st.stop()

    df = df[['SMILES_STANDARDIZED', 'CLASS']].dropna()

    # Size control
    if len(df) > MAX_MOLECULES:
        st.warning(f"Dataset too large ({len(df)}). Subsampling to {MAX_MOLECULES}")
        df = df.sample(MAX_MOLECULES, random_state=42)

    # =====================
    # FINGERPRINTS
    # =====================
    fps, valid_idx, invalid = compute_fps(df['SMILES_STANDARDIZED'], radius, nbits)

    st.write(f"Valid molecules: {len(valid_idx)}")
    st.write(f"Invalid SMILES: {invalid}")

    df = df.iloc[valid_idx].reset_index(drop=True)
    classes = df.CLASS.values

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

    # =====================
    # STATS + TEST
    # =====================
    stat, pval = mannwhitneyu(v11, v10, alternative="greater")

    st.subheader("Statistical test")
    st.write(f"Mann–Whitney p-value (1-1 > 1-0): {pval:.3e}")

    # =====================
    # HEATMAP
    # =====================
    st.subheader("Clustered Heatmap")

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
        figsize=(8,8)
    )

    st.pyplot(fig)

    # =====================
    # KDE
    # =====================
    st.subheader("Similarity distributions")

    fig2, ax = plt.subplots()

    sns.kdeplot(v11, label="1-1", fill=True, ax=ax)
    sns.kdeplot(v00, label="0-0", fill=True, ax=ax)
    sns.kdeplot(v10, label="1-0", fill=True, ax=ax)

    ax.legend()
    st.pyplot(fig2)

    # =====================
    # UMAP
    # =====================
    st.subheader("UMAP projection")

    fp_array = np.array([list(fp) for fp in fps])

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="jaccard")
    emb = reducer.fit_transform(fp_array)

    fig3, ax = plt.subplots()
    ax.scatter(emb[:,0], emb[:,1], c=classes)
    st.pyplot(fig3)

    # =====================
    # METRICS TABLE
    # =====================
    results = pd.DataFrame({
        "group": ["class1", "class0", "inter"],
        "mean": [np.mean(v11), np.mean(v00), np.mean(v10)]
    })

    st.subheader("Metrics")
    st.dataframe(results)

    # =====================
    # DOWNLOAD
    # =====================
    csv = results.to_csv(index=False).encode()
    st.download_button("Download metrics", csv, "metrics.csv")

    # =====================
    # INTERPRETATION
    # =====================
    st.subheader("Quick interpretation")

    delta = np.mean(v11) - np.mean(v10)

    if delta < 0.02:
        st.warning("Very weak structural separation")
    elif delta < 0.05:
        st.info("Moderate separation")
    else:
        st.success("Strong structural enrichment")
