import streamlit as st
import pandas as pd
import numpy as np
import os

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

st.title("Structural Diversity Analysis")

# =========================
# DATA LOADING
# =========================
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.info("No file uploaded → loading example_dataset.csv")
    if os.path.exists("example_dataset.csv"):
        df = pd.read_csv("example_dataset.csv")
    else:
        st.error("No dataset found")
        st.stop()

# =========================
# VALIDATION
# =========================
required = {"SMILES_STANDARDIZED", "CLASS"}
if not required.issubset(df.columns):
    st.error("Missing required columns")
    st.stop()

df = df.dropna(subset=["SMILES_STANDARDIZED", "CLASS"])

st.subheader("Dataset overview (before filtering)")
st.write(f"Total molecules: {len(df)}")
st.write(df["CLASS"].value_counts().rename("count"))

# =========================
# SMILES VALIDATION
# =========================
def validate(df):
    valid = []
    invalid = 0

    for i, smi in enumerate(df.SMILES_STANDARDIZED):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            valid.append(i)
        else:
            invalid += 1

    return valid, invalid

valid_idx, invalid = validate(df)

df = df.iloc[valid_idx].reset_index(drop=True)

st.write(f"Valid SMILES: {len(df)}")
st.write(f"Invalid SMILES removed: {invalid}")

st.write("Class distribution (after filtering)")
st.write(df["CLASS"].value_counts().rename("count"))

if len(df) > MAX_MOLECULES:
    st.warning(f"Subsampling to {MAX_MOLECULES}")
    df = df.sample(MAX_MOLECULES, random_state=42)

smiles = df.SMILES_STANDARDIZED.tolist()
classes = df.CLASS.values

# =========================
# FINGERPRINTS
# =========================
radius = st.sidebar.slider("Radius", 1, 4, 2)
nbits = st.sidebar.selectbox("FP length", [512, 1024, 2048, 4096], 2)

def fps(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    return [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]

fps = fps(smiles)

def sim_matrix(fps):
    n = len(fps)
    sim = np.zeros((n, n))
    for i in range(n):
        sim[i, :] = BulkTanimotoSimilarity(fps[i], fps)
    return sim

sim = sim_matrix(fps)

idx1 = np.where(classes == 1)[0]
idx0 = np.where(classes == 0)[0]

sim11 = sim[np.ix_(idx1, idx1)]
sim00 = sim[np.ix_(idx0, idx0)]
sim10 = sim[np.ix_(idx1, idx0)]

v11 = sim11[np.triu_indices_from(sim11, 1)]
v00 = sim00[np.triu_indices_from(sim00, 1)]
v10 = sim10.flatten()

# =========================================================
# HEATMAP + EXPLANATION
# =========================================================
st.subheader("Clustered Heatmap")

st.markdown("""
This heatmap represents **pairwise Tanimoto similarity between molecules**,
reordered using hierarchical clustering.

### How to interpret:
- Bright blocks → structurally similar compounds
- Dendrogram ordering → groups similar molecules together
- Color strip → class membership (red = 1, blue = 0)

⚠️ Important: visual clusters do not imply biological significance.
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

# =========================================================
# STATISTICAL TEST (POST HEATMAP)
# =========================================================
st.subheader("Statistical test (Mann–Whitney U)")

st.markdown("""
We test whether **intra-class similarity (1–1)** is greater than **inter-class similarity (1–0)**.

⚠️ This test only evaluates distributions of similarity values.
It does NOT imply causal or biological separation.
""")

_, pval = mannwhitneyu(v11, v10, alternative="greater")
st.write(f"p-value: {pval:.3e}")

# =========================================================
# SIMILARITY DISTRIBUTIONS
# =========================================================
st.subheader("Similarity distributions")

fig, ax = plt.subplots()

sns.kdeplot(v11, label="1–1", fill=True, ax=ax)
sns.kdeplot(v00, label="0–0", fill=True, ax=ax)
sns.kdeplot(v10, label="1–0", fill=True, ax=ax)

ax.legend()
st.pyplot(fig)

# =========================================================
# UMAP
# =========================================================
st.subheader("UMAP projection")

fp_array = np.array([list(fp) for fp in fps])
emb = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="jaccard").fit_transform(fp_array)

fig, ax = plt.subplots()
ax.scatter(emb[:, 0], emb[:, 1], c=classes, cmap="coolwarm", s=20)
st.pyplot(fig)

# =========================================================
# SCAFFOLDS (2D STRUCTURES)
# =========================================================
st.subheader("Scaffold diversity (2D structures)")

def get_scaffolds(smiles):
    scaffolds = []
    mols = []

    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            scaf = MurckoScaffold.GetScaffoldForMol(mol)
            scaffolds.append(Chem.MolToSmiles(scaf))
            mols.append(scaf)

    return scaffolds, mols

scaf_all, _ = get_scaffolds(smiles)
scaf1, mols1 = get_scaffolds(df[df.CLASS == 1].SMILES_STANDARDIZED)
scaf0, mols0 = get_scaffolds(df[df.CLASS == 0].SMILES_STANDARDIZED)

def draw_top(scafs, mols, title):
    counts = Counter(scafs).most_common(5)

    mol_list = []
    labels = []

    for smi, count in counts:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mol_list.append(mol)
            labels.append(f"n={count}")

    img = Draw.MolsToGridImage(
        mol_list,
        molsPerRow=5,
        legends=labels,
        subImgSize=(200, 200)
    )
    return img

col1, col2, col3 = st.columns(3)

with col1:
    st.write("All")
    st.image(draw_top(scaf_all, None, "All"))

with col2:
    st.write("Class 1")
    st.image(draw_top(scaf1, None, "Class 1"))

with col3:
    st.write("Class 0")
    st.image(draw_top(scaf0, None, "Class 0"))

# =========================================================
# PHYSICOCHEMICAL PROPERTIES
# =========================================================
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

fig, axes = plt.subplots(1, 5, figsize=(16, 3))

for i in range(5):
    sns.boxplot(x=classes, y=props[:, i], ax=axes[i], palette={0:"#1f77b4", 1:"#d62728"})
    axes[i].set_title(props_names[i])

st.pyplot(fig)

# =========================================================
# STAT TEST FOR PHYSICOCHEM
# =========================================================
st.subheader("Physicochemical statistical tests")

results = []

for i, name in enumerate(props_names):
    p0 = props[classes == 0][:, i]
    p1 = props[classes == 1][:, i]

    stat, p = mannwhitneyu(p1, p0, alternative="two-sided")

    results.append([name, np.mean(p1), np.mean(p0), p])

res_df = pd.DataFrame(results, columns=["Property", "Mean Class 1", "Mean Class 0", "p-value"])

st.dataframe(res_df)
