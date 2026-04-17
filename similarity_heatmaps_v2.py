import streamlit as st
import pandas as pd
import numpy as np
import os
from io import BytesIO
import base64

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Draw

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

st.set_page_config(page_title="Similarity Heatmaps", layout="wide")

st.title("Structural Diversity Analysis Platform")

paper_mode = st.toggle("📄 Paper export mode", value=False)

# =========================
# LOAD DATA
# =========================
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Using example_dataset.csv")
    df = pd.read_csv("example_dataset.csv")

# =========================
# VALIDATION
# =========================
required = {"SMILES_STANDARDIZED", "CLASS"}
if not required.issubset(df.columns):
    st.error("Missing required columns")
    st.stop()

df = df.dropna(subset=["SMILES_STANDARDIZED", "CLASS"])

st.subheader("Dataset overview")
st.write(df["CLASS"].value_counts())

# =========================
# SMILES VALIDATION
# =========================
valid_idx = []
invalid = 0

for i, smi in enumerate(df.SMILES_STANDARDIZED):
    mol = Chem.MolFromSmiles(smi)
    if mol:
        valid_idx.append(i)
    else:
        invalid += 1

df = df.iloc[valid_idx].reset_index(drop=True)

st.write(f"Valid SMILES: {len(df)}")
st.write(f"Invalid SMILES removed: {invalid}")

df["CLASS"] = df["CLASS"].astype(int)

classes = df.CLASS.values
smiles = df.SMILES_STANDARDIZED.tolist()

if len(df) > MAX_MOLECULES:
    st.warning(f"Subsampling to {MAX_MOLECULES}")
    df = df.sample(MAX_MOLECULES, random_state=42)

# =========================
# FINGERPRINTS
# =========================
radius = st.sidebar.slider("Radius", 1, 4, 2)
nbits = st.sidebar.selectbox("FP length", [512, 1024, 2048, 4096], 2)

def get_fps(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    return [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]

fps = get_fps(smiles)

def sim_matrix(fps):
    n = len(fps)
    sim = np.zeros((n, n))
    for i in range(n):
        sim[i, :] = BulkTanimotoSimilarity(fps[i], fps)
    return sim

sim = sim_matrix(fps)

# =========================
# HEATMAP (UI SMALL + EXPORT LARGE)
# =========================
st.subheader("Clustered Heatmap")

st.markdown("""
Pairwise Tanimoto similarity clustered hierarchically.

- blocks = structural similarity groups  
- dendrogram = clustering order  
- colors = class labels  
""")

dist = 1 - sim
link = linkage(squareform(dist, checks=False), method="average")

colors = pd.Series(classes).map({0:"#1f77b4", 1:"#d62728"}).values

# SMALL FIGURE FOR UI
fig = sns.clustermap(
    sim,
    row_linkage=link,
    col_linkage=link,
    row_colors=colors,
    col_colors=colors,
    cmap="viridis",
    xticklabels=False,
    yticklabels=False,
    figsize=(6, 6)
)

st.pyplot(fig)

# =========================
# EXPORT HEATMAP (HIGH RES)
# =========================
buf = BytesIO()
fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
buf.seek(0)

st.download_button(
    "⬇️ Download heatmap (300 DPI PNG)",
    data=buf,
    file_name="heatmap.png",
    mime="image/png"
)

# =========================
# STAT TEST
# =========================
st.subheader("Statistical test (Mann–Whitney U)")

idx1 = np.where(classes == 1)[0]
idx0 = np.where(classes == 0)[0]

sim11 = sim[np.ix_(idx1, idx1)]
sim10 = sim[np.ix_(idx1, idx0)]

v11 = sim11[np.triu_indices_from(sim11, 1)]
v10 = sim10.flatten()

_, pval = mannwhitneyu(v11, v10, alternative="greater")

st.write(f"p-value: {pval:.3e}")

# =========================
# DISTRIBUTIONS
# =========================
st.subheader("Similarity distributions")

fig2, ax = plt.subplots()

sns.kdeplot(v11, label="1–1", fill=True, ax=ax)
sns.kdeplot(v10, label="1–0", fill=True, ax=ax)

ax.legend()
st.pyplot(fig2)

# =========================
# UMAP
# =========================
st.subheader("UMAP projection")

fp_array = np.array([list(fp) for fp in fps])
emb = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="jaccard").fit_transform(fp_array)

fig3, ax = plt.subplots(figsize=(6, 5))
ax.scatter(emb[:, 0], emb[:, 1], c=classes, cmap="coolwarm", s=20)
st.pyplot(fig3)

# =========================
# SCAFFOLDS (FIXED, NO SVG)
# =========================
st.subheader("Scaffold diversity (Murcko scaffolds)")

def get_scaffolds(smiles):
    counts = Counter()

    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            scaf = MurckoScaffold.GetScaffoldForMol(mol)
            counts[Chem.MolToSmiles(scaf)] += 1

    top = counts.most_common(5)

    mols = [Chem.MolFromSmiles(s) for s, _ in top]
    labels = [f"n={c}" for _, c in top]

    return mols, labels

def draw_scaffolds(mols, labels):
    return Draw.MolsToGridImage(
        mols,
        legends=labels,
        molsPerRow=1,
        subImgSize=(400, 400),
        useSVG=False
    )

m1, l1 = get_scaffolds(df[df.CLASS == 1].SMILES_STANDARDIZED)
m0, l0 = get_scaffolds(df[df.CLASS == 0].SMILES_STANDARDIZED)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Class 1")
    st.image(draw_scaffolds(m1, l1))

with col2:
    st.markdown("### Class 0")
    st.image(draw_scaffolds(m0, l0))

# =========================
# PHYSICOCHEMICAL PROPERTIES
# =========================
st.subheader("Physicochemical properties")

props_names = ["MW", "LogP", "HBD", "HBA", "RotB"]

def calc_props(smiles):
    out = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            out.append([
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol)
            ])
    return np.array(out)

props = calc_props(smiles)

fig4, axes = plt.subplots(1, 5, figsize=(16, 3))

for i in range(5):
    axes[i].boxplot([
        props[classes == 0][:, i],
        props[classes == 1][:, i]
    ])
    axes[i].set_title(props_names[i])
    axes[i].set_xticklabels(["0", "1"])

st.pyplot(fig4)

# =========================
# STAT SUMMARY TABLE
# =========================
st.subheader("Physicochemical statistical summary")

rows = []

for i, name in enumerate(props_names):
    p0 = props[classes == 0][:, i]
    p1 = props[classes == 1][:, i]

    _, p = mannwhitneyu(p1, p0)

    rows.append([name, np.mean(p0), np.mean(p1), p])

st.dataframe(pd.DataFrame(rows, columns=["Property", "Mean 0", "Mean 1", "p-value"]))
