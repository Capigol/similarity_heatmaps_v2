import streamlit as st
import pandas as pd
import numpy as np
import os

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
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
# LOAD DATA
# =========================
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("No file uploaded → loading example_dataset.csv")
    if os.path.exists("example_dataset.csv"):
        df = pd.read_csv("example_dataset.csv")
    else:
        st.error("No dataset found.")
        st.stop()

# =========================
# BASIC VALIDATION (MOVED TO TOP)
# =========================
required = {"SMILES_STANDARDIZED", "CLASS"}

if not required.issubset(df.columns):
    st.error("Missing required columns: SMILES_STANDARDIZED, CLASS")
    st.stop()

df = df.dropna(subset=["SMILES_STANDARDIZED", "CLASS"])

# CLASS distribution BEFORE filtering
st.subheader("Dataset overview (before filtering)")
st.write(f"Total molecules: {len(df)}")
st.write(df["CLASS"].value_counts().rename("count"))

# =========================
# SMILES VALIDATION (IMPORTANT)
# =========================
def validate_smiles(df):
    valid_idx = []
    invalid = 0

    for i, smi in enumerate(df.SMILES_STANDARDIZED):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            valid_idx.append(i)
        else:
            invalid += 1

    return valid_idx, invalid

valid_idx, invalid = validate_smiles(df)

df = df.iloc[valid_idx].reset_index(drop=True)

st.write(f"Valid SMILES: {len(df)}")
st.write(f"Invalid SMILES removed: {invalid}")

st.write("Class distribution (after filtering)")
st.write(df["CLASS"].value_counts().rename("count"))

# Subsample
if len(df) > MAX_MOLECULES:
    st.warning(f"Subsampling to {MAX_MOLECULES}")
    df = df.sample(MAX_MOLECULES, random_state=42)

smiles = df.SMILES_STANDARDIZED.tolist()
classes = df.CLASS.values

# =========================
# FP + SIMILARITY
# =========================
radius = st.sidebar.slider("Radius", 1, 4, 2)
nbits = st.sidebar.selectbox("FP length", [512, 1024, 2048, 4096], 2)

def compute_fps(smiles, radius, nbits):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    return [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]

fps = compute_fps(smiles, radius, nbits)

def tanimoto_matrix(fps):
    n = len(fps)
    sim = np.zeros((n, n))
    for i in range(n):
        sim[i, :] = BulkTanimotoSimilarity(fps[i], fps)
    return sim

sim = tanimoto_matrix(fps)

idx1 = np.where(classes == 1)[0]
idx0 = np.where(classes == 0)[0]

sim11 = sim[np.ix_(idx1, idx1)]
sim00 = sim[np.ix_(idx0, idx0)]
sim10 = sim[np.ix_(idx1, idx0)]

v11 = sim11[np.triu_indices_from(sim11, k=1)]
v00 = sim00[np.triu_indices_from(sim00, k=1)]
v10 = sim10.flatten()

# =========================
# HEATMAP
# =========================
st.subheader("Clustered Heatmap")

st.markdown("""
Pairwise Tanimoto similarity clustered by hierarchical clustering.

Interpretation:
- blocks = structural neighborhoods
- color stripes = class labels
- separation suggests structural enrichment
""")

dist = 1 - sim
link = linkage(squareform(dist, checks=False), method="average")

colors = {0: "#1f77b4", 1: "#d62728"}
row_colors = pd.Series(classes).map(colors).to_numpy()

sns.clustermap(
    sim,
    row_linkage=link,
    col_linkage=link,
    row_colors=row_colors,
    col_colors=row_colors,
    cmap="viridis",
    xticklabels=False,
    yticklabels=False
)

st.pyplot(plt)

# =========================
# STATISTICAL TEST (MOVED HERE)
# =========================
st.subheader("Statistical test (Mann–Whitney U)")

st.markdown("""
This test evaluates whether intra-class similarity (Class 1–1)
is statistically higher than inter-class similarity (Class 1–0).

⚠️ Interpretation:
A small p-value suggests a tendency toward structural enrichment,
but does NOT imply strong or biologically meaningful separation.
""")

_, pval = mannwhitneyu(v11, v10, alternative="greater")
st.write(f"p-value: {pval:.3e}")

# =========================
# DISTRIBUTIONS
# =========================
st.subheader("Similarity distributions")

fig, ax = plt.subplots()
sns.kdeplot(v11, label="1-1", fill=True, ax=ax)
sns.kdeplot(v00, label="0-0", fill=True, ax=ax)
sns.kdeplot(v10, label="1-0", fill=True, ax=ax)
ax.legend()
st.pyplot(fig)

# =========================
# UMAP
# =========================
st.subheader("UMAP projection")

fp_array = np.array([list(fp) for fp in fps])

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="jaccard")
emb = reducer.fit_transform(fp_array)

fig, ax = plt.subplots()
ax.scatter(emb[:, 0], emb[:, 1], c=classes, cmap="coolwarm", s=20)
st.pyplot(fig)

# =========================
# SCAFFOLDS (2D PLOT)
# =========================
st.subheader("Scaffold diversity (Murcko scaffolds)")

def get_scaffolds(smiles):
    scaffolds = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            scaf = MurckoScaffold.GetScaffoldForMol(mol)
            scaffolds.append(Chem.MolToSmiles(scaf))
    return Counter(scaffolds)

scaf1 = get_scaffolds(df[df.CLASS == 1].SMILES_STANDARDIZED)
scaf0 = get_scaffolds(df[df.CLASS == 0].SMILES_STANDARDIZED)

def plot_top(scaf, title):
    top = scaf.most_common(5)
    names = [f"S{i+1}" for i in range(len(top))]
    counts = [c for _, c in top]

    fig, ax = plt.subplots()
    ax.barh(names, counts)
    ax.set_title(title)
    return fig

col1, col2 = st.columns(2)

with col1:
    st.pyplot(plot_top(scaf1, "Class 1 top scaffolds"))

with col2:
    st.pyplot(plot_top(scaf0, "Class 0 top scaffolds"))

# =========================
# PHYSICOCHEMICAL PROPERTIES
# =========================
st.subheader("Physicochemical properties")

def calc_props(smiles):
    props = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            props.append([
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol)
            ])
    return np.array(props)

props = calc_props(smiles)

labels = ["MW", "LogP", "HBD", "HBA", "RotB"]

fig, axes = plt.subplots(1, 5, figsize=(15, 3))

for i in range(5):
    sns.boxplot(x=classes, y=props[:, i], ax=axes[i])
    axes[i].set_title(labels[i])

st.pyplot(fig)
