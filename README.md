# PockLigGPT

PockLigGPT is a pocket-conditioned molecular generation framework based on GPT architectures and reinforcement learning (RL) for structure-based drug design.

It supports multiple workflows:

1. **Reproduce training** (pretraining + finetuning) from tokenized datasets
2. **Use pretrained checkpoints** and run RL
3. **Condition RL with a real pocket** using sequence + ProtT5 embeddings (`.npy`) + receptor + docking coordinates

---

## 🚀 Installation

### Option A: pip (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Option B: conda

```bash
conda env create -f environment.yml
conda activate pockliggpt
pip install -e .
```

---

## ⚡ Quickstart

```bash
pip install -e .
python scripts/train_ppo.py --config config/rl/sequence_add.yaml
```

---

## 📂 Workflow

### 1) Prepare datasets

Datasets are **not included**.

Place raw data in:

```bash
datasets/raw/
```

Typical datasets:

* ChEMBL
* ZINC20
* CrossDocked

---

### 2) Tokenization

```bash
python scripts/tokenize_dataset.py --config config/tokenization/chembl.yaml
```

Outputs:

```bash
datasets/processed/*.bin
```

---

### 3) Training

```bash
python scripts/train.py --config config/training/pretrain/zinc_20_sequence.yaml
```

---

### 4) Pretrained checkpoints

Download from Hugging Face:

👉 https://huggingface.co/pablovp8/pockliggpt-models

Or:

```bash
python -m huggingface_hub download pablovp8/pockliggpt-models \
  --repo-type model \
  --local-dir checkpoints/pockliggpt
```

---

### 5) RL with pocket conditioning

```bash
python scripts/train_ppo.py --config config/rl/sequence_add.yaml
```

---

## 🧬 Pocket embeddings

Use:

```bash
notebooks/prott5_pocket_pipeline_simple_en.ipynb
```

Outputs:

* pocket sequence
* `.npy` embeddings

---

## ⚙️ Docking setup

Docking is required for RL reward.

### Automatic setup (recommended)

```bash
bash scripts/setup_docking.sh
```

This will:

* download `docking_vina`
* setup AutoGrow structure
* guide MGLTools installation

---

### Manual setup

1. Install:

* AutoDock Vina
* Open Babel

2. Download MGLTools:

👉 https://ccsb.scripps.edu/mgltools/

```bash
tar -zxvf mgltools_*.tar.gz
cd mgltools_*
./install.sh
```

---

### Configure docking

Edit:

```bash
config/docking/vars_mgltools.json
```

Set:

* receptor path
* center_x, center_y, center_z
* size_x, size_y, size_z
* MGLTools paths

---

## 📁 Project structure

```bash
config/
datasets/
pockliggpt/
scripts/
notebooks/
tests/
```

---

## 🔗 External dependencies

* AutoDock Vina
* AutoGrow
* MGLTools

These are **not included** and must be installed separately.

---

## ⚠️ Status

Actively developed.
Stable for inference and RL workflows.

---

## 📜 License

MIT License

---
