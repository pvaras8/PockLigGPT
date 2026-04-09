# PockLigGPT

PockLigGPT is a pocket-conditioned molecular generation framework based on GPT architectures and reinforcement learning for structure-based drug design.

The model generates molecules conditioned on protein binding pockets and optimizes them using docking-based rewards.

---

## Workflow

The typical workflow is:

### 1. Extract pocket sequence & embeddings

Use the provided notebook:

notebooks/pocket_to_prott5_embeddings.ipynb

This notebook allows you to:

- Load a PDB file
- Select a binding pocket (ligand-based or coordinates)
- Extract the amino acid sequence of the pocket
- Compute ProtT5 residue-level embeddings

The output is saved as `.npy` files.

---

### 2. Load pretrained model

Download pretrained weights from:

https://huggingface.co/<your_user>/pockliggpt-models

---

### 3. Run molecular generation

python scripts/sample.py --config config/sampling/sample.yaml

---

## Features

- Pocket-conditioned molecular generation
- SELFIES-based molecular representation
- GPT-style autoregressive model
- Reinforcement learning optimization
- Docking-based reward functions
- ProtT5 protein embeddings

---

## Status

⚠️ This repository is currently being prepared for public release.

Code, documentation, and reproducibility scripts are being finalized.

---

## Planned contents

- Training pipeline
- Molecular generation scripts
- Docking reward integration
- Example datasets
- Reproducibility instructions

---

## Citation

If you use this code in your research, please cite the associated paper:
