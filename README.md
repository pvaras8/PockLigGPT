# PockLigGPT

PockLigGPT is a pocket-conditioned molecular generation framework based
on GPT architectures and reinforcement learning (RL) for structure-based
drug design.

------------------------------------------------------------------------

## 🌐 Online access

PockLigGPT is also available through a research collaboration interface:

👉 https://pockliggpt.streamlit.app

Researchers and industry partners can submit target proteins (PDB) and
project descriptions to request molecule generation or full
computational studies.

------------------------------------------------------------------------

## 🔗 Model Weights

Download released checkpoints from Hugging Face:

👉 https://huggingface.co/pablovp8/PockLigGPT

Pretrained and fine-tuned checkpoints are provided for direct use.

------------------------------------------------------------------------

PockLigGPT supports multiple workflows:

1.  Reproduce training (pretraining + finetuning)\
2.  Use pretrained checkpoints and run RL\
3.  Condition RL with real pocket inputs

------------------------------------------------------------------------

## 🚀 Installation

``` bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

------------------------------------------------------------------------

## ⚡ Quickstart

``` bash
python scripts/train_ppo.py --config config/rl/sequence_add.yaml
```

------------------------------------------------------------------------

## 📂 Workflow

### 1) Prepare datasets

datasets/raw/

### 2) Tokenization

python scripts/tokenize_dataset.py --config ...

### 3) Training

python scripts/train.py --config ...

### 4) Pretrained checkpoints

``` bash
python -m huggingface_hub download pablovp8/PockLigGPT \
  --repo-type model \
  --local-dir checkpoints/pockliggpt
```

### 5) RL

``` bash
python scripts/train_ppo.py --config config/rl/sequence_add.yaml
```

------------------------------------------------------------------------

## 🧬 Pocket Embeddings

To condition generation on protein structure, pocket embeddings are
required.

Use:

``` bash
notebooks/prott5_pocket_pipeline_simple_en.ipynb
```

This pipeline generates: - pocket amino-acid sequence\
- ProtT5 residue embeddings (`.npy`)

These are required inputs for pocket-conditioned RL.

------------------------------------------------------------------------

## ⚙️ Docking Setup

Docking is used as the reward signal during RL optimization.

### 1) Download docking pipeline

``` bash
bash scripts/setup_docking.sh
```

This downloads the `docking_vina/` folder containing the prepared
docking workflow.

------------------------------------------------------------------------

### 2) Install MGLTools

Download from:

👉 https://ccsb.scripps.edu/mgltools/

Example:

``` bash
tar -zxvf mgltools_*.tar.gz
cd mgltools_x86_64Linux2_1.5.6
./install.sh
cd ..
```

Typical HPC path:

    /LUSTRE/users/<user>/

------------------------------------------------------------------------

### 3) Configure docking

Edit:

``` bash
config/docking/vars_mgltools.json
```

Set: - receptor path\
- center_x, center_y, center_z\
- size_x, size_y, size_z\
- MGLTools paths

------------------------------------------------------------------------

## ✅ Minimal Checklist

Before running RL:

-   datasets ready\
-   tokenizer `.pkl` available\
-   ProtT5 embeddings generated\
-   checkpoint path valid\
-   `docking_vina/` downloaded\
-   MGLTools installed\
-   docking config completed

------------------------------------------------------------------------

## 📁 Project Structure

``` bash
config/
datasets/
pockliggpt/
scripts/
notebooks/
tests/
```

------------------------------------------------------------------------

## 🔗 External Dependencies

-   MGLTools (ligand and receptor preparation for docking)

------------------------------------------------------------------------

## ⚠️ Status

Actively developed.\
Stable for inference and RL workflows.

------------------------------------------------------------------------

## 📜 License

MIT License
