# PockLigGPT

PockLigGPT is a pocket-conditioned molecular generation framework based on GPT architectures and reinforcement learning (RL) for structure-based drug design.

It supports multiple workflows:

1. **Reproduce training** (pretraining + finetuning) from tokenized datasets
2. **Use pretrained checkpoints** (e.g. from Hugging Face) and run RL
3. **Condition RL with a real pocket** using sequence + ProtT5 embeddings (`.npy`) + receptor + docking coordinates

---

## 🚀 Installation

### Option A: pip (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

### Option B: conda

```bash
conda env create -f environment.yml
conda activate pockliggpt
python -m pip install -e .
```

### Sanity check

```bash
python -c "import pockliggpt; print(pockliggpt.__version__)"
```

---

## ⚡ Quickstart (minimal)

```bash
pip install -e .
python scripts/train_ppo.py --config config/rl/sequence_add.yaml
```

---

## 📂 Workflow (end-to-end)

### 1) Prepare datasets

Datasets are **not included**.

* See `datasets/README.md`
* Place raw data under `datasets/raw/...`

Typical datasets:

* ChEMBL
* ZINC20
* CrossDocked

---

### 2) Tokenization

```bash
python scripts/tokenize_dataset.py --config config/tokenization/chembl.yaml
python scripts/tokenize_dataset.py --config config/tokenization/zinc20.yaml
python scripts/tokenize_dataset.py --config config/tokenization/crossdocked.yaml
```

Outputs:

* `datasets/processed/*.bin`

> Update dataset paths inside YAML configs.

Tokenizer metadata (`meta_*.pkl`) must be placed in:

```bash
datasets/tokenizer/
```

---

### 3) Training

```bash
python scripts/train.py --config config/training/pretrain/zinc_20_sequence.yaml
python scripts/train.py --config config/training/finetune_1/chembl_sequence.yaml
python scripts/train.py --config config/training/finetune_2/crossdocked_sequence.yaml
```

Supports multi-GPU via `torchrun`.

---

### 4) Use pretrained checkpoints

```bash
python -m huggingface_hub download <HF_USER>/<HF_REPO> \
  --repo-type model \
  --local-dir checkpoints/pockliggpt
```

Then set in configs:

* `model.checkpoint_path`
* `training.init_ckpt_path`

---

### 5) RL with pocket conditioning

```bash
python scripts/train_ppo.py --config config/rl/sequence_add.yaml
```

#### A) Extract pocket embeddings

Use:

```bash
notebooks/prott5_pocket_pipeline_simple_en.ipynb
```

Outputs:

* pocket sequence
* `.npy` embeddings

---

#### B) Configure RL

Set in YAML:

* `conditioning.pocket_str`
* `conditioning.pocket_emb_path`
* `model.checkpoint_path`
* `tokenizer.meta_path`

---

#### C) Configure docking

Edit:

```bash
config/docking/vars_meeko.json
# or
config/docking/vars_mgltools.json
```

Required:

* receptor path
* center_x / center_y / center_z
* size_x / size_y / size_z

---

## ✅ Minimal checklist

* datasets in place
* tokenizer `.pkl` available
* `.npy` embeddings generated
* checkpoint path valid
* docking config filled

---

## 🧠 Features

* Pocket-conditioned generation
* SELFIES molecular representation
* GPT-based autoregressive modeling
* PPO reinforcement learning
* Docking-based reward
* ProtT5 protein embeddings

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

This project relies on external tools:

* AutoDock Vina
* Meeko
* AutoGrow

These are **not included** and must be installed separately.

Users must comply with their respective licenses.

---

## ⚠️ Status

Actively developed.
Stable for inference and RL workflows.

---

## 📜 License

MIT License

---

## 📖 Citation


