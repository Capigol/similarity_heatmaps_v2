# Structural Diversity Analysis App

A Streamlit-based application for analyzing structural diversity in molecular datasets using Morgan fingerprints and Tanimoto similarity.

This tool is designed for cheminformatics workflows, allowing users to explore intra- and inter-class similarity, scaffold diversity, and structural clustering in an interactive and reproducible way.

---

## 🚀 Features

- Upload CSV datasets with molecular structures
- Configurable Morgan fingerprints (radius and bit length)
- Tanimoto similarity matrix computation
- Clustered heatmap visualization
- Similarity distribution plots (KDE)
- UMAP projection of chemical space
- Nearest neighbor analysis
- Scaffold diversity metrics (Murcko scaffolds)
- Statistical comparison (Mann–Whitney test)
- Exportable metrics

---

## 📥 Input format

The input CSV file must contain at least the following columns:

- `SMILES_STANDARDIZED`: standardized SMILES strings
- `CLASS`: binary class label (e.g., 0 = inactive, 1 = active)

Example:

| SMILES_STANDARDIZED | CLASS |
|---------------------|------|
| CC(=O)Oc1ccccc1C(=O)O | 1 |
| CCO | 0 |

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/capigol/similarity_heatmaps_v2
cd your-repo
