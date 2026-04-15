# PockLigGPT

PockLigGPT is a pocket-conditioned molecular generation framework based on GPT architectures and reinforcement learning (RL) for structure-based drug design.

## 🌐 Online access

PockLigGPT is also available through a research collaboration interface:

👉 https://pockliggpt.streamlit.app

Researchers and industry partners can submit target proteins (PDB) and project descriptions to request molecule generation or full computational studies.

---

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

### Sanity check

```bash
python -c "import pockliggpt; print(pockliggpt.__version__)"
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
- ChEMBL
- ZINC20
- CrossDocked

---

### 2) Tokenization

```bash
python scripts/tokenize_dataset.py --config config/tokenization/chembl.yaml
python scripts/tokenize_dataset.py --config config/tokenization/zinc20.yaml
python scripts/tokenize_dataset.py --config config/tokenization/crossdocked.yaml
```

Outputs:

```bash
datasets/processed/*.bin
```

Tokenizer metadata (`meta_*.pkl`) should be placed in:

```bash
datasets/tokenizer/
```

> Update dataset paths inside YAML configs to match your local machine.

---

### 3) Training

```bash
python scripts/train.py --config config/training/pretrain/zinc_20_sequence.yaml
python scripts/train.py --config config/training/finetune_1/chembl_sequence.yaml
python scripts/train.py --config config/training/finetune_2/crossdocked_sequence.yaml
```

Supports multi-GPU via `torchrun`.

---

<!-- ### 4) Pretrained checkpoints

Download released checkpoints from Hugging Face:

👉 https://huggingface.co/pablovp8/pockliggpt-models

Or with:

```bash
python -m huggingface_hub download pablovp8/pockliggpt-models \
  --repo-type model \
  --local-dir checkpoints/pockliggpt
```

Then set the checkpoint path in your config files. -->

---

### 4) RL with pocket conditioning

```bash
python scripts/train_ppo.py --config config/rl/sequence_add.yaml
```

Alternative configs:
- `config/rl/sequence.yaml`
- `config/rl/sequence_add.yaml`
- `config/rl/cross.yaml`

---

## 🧬 Pocket embeddings

Use:

```bash
notebooks/prott5_pocket_pipeline_simple_en.ipynb
```

This notebook exports:
- pocket amino-acid sequence
- ProtT5 residue embeddings (`.npy`)

These outputs are required for pocket-conditioned RL.

---

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

Datasets are **not included**.

Place raw data in:

```bash
datasets/raw/
```

Typical datasets:
- ChEMBL
- ZINC20
- CrossDocked

---

------------------------------------------------------------------------

### 2) Tokenization

``` bash
python scripts/tokenize_dataset.py --config config/tokenization/chembl.yaml
python scripts/tokenize_dataset.py --config config/tokenization/zinc20.yaml
python scripts/tokenize_dataset.py --config config/tokenization/crossdocked.yaml
```

------------------------------------------------------------------------

### 3) Training

``` bash
python scripts/train.py --config config/training/pretrain/zinc_20_sequence.yaml
python scripts/train.py --config config/training/finetune_1/chembl_sequence.yaml
python scripts/train.py --config config/training/finetune_2/crossdocked_sequence.yaml
```

------------------------------------------------------------------------

### 4) Pretrained checkpoints

``` bash
python -m huggingface_hub download pablovp8/PockLigGPT \
  --repo-type model \
  --local-dir checkpoints/pockliggpt
```

------------------------------------------------------------------------

### 5) RL with pocket conditioning

``` bash
python scripts/train_ppo.py --config config/rl/sequence_add.yaml
```

------------------------------------------------------------------------

## 🧬 Pocket Embeddings

Use:

``` bash
notebooks/prott5_pocket_pipeline_simple_en.ipynb
```

This generates: - pocket amino-acid sequence\
- ProtT5 residue embeddings (`.npy`)

------------------------------------------------------------------------

## ⚙️ Docking setup

Docking is required for RL reward.

### Automatic download of `docking_vina`

```bash
bash scripts/setup_docking.sh
```

This downloads and unpacks the prepared `docking_vina/` folder used by the project.

### What is included in `docking_vina`?

The downloaded `docking_vina/` bundle contains the prepared docking workflow used in this project, including the required AutoGrow-related files and folder structure.

### MGLTools installation

MGLTools must still be installed **manually** and **outside the repository**.

Download MGLTools from:

👉 https://ccsb.scripps.edu/mgltools/

Then install it, for example:

```bash
tar -zxvf mgltools_*.tar.gz
cd mgltools_x86_64Linux2_1.5.6
./install.sh
cd ..
```

This is typically installed in a user directory, for example on HPC systems:

```bash
/LUSTRE/users/<user>/
```
### Configure docking

Edit:

```bash
config/docking/vars_mgltools.json
```

Set:
- receptor path
- `center_x`, `center_y`, `center_z`
- `size_x`, `size_y`, `size_z`


---

## ✅ Minimal checklist

Before running RL with docking reward, make sure that:

- datasets are in place
- tokenizer `.pkl` is available
- `.npy` embeddings are generated
- checkpoint path is valid
- `docking_vina/` has been downloaded
- MGLTools is installed outside the repo
- docking config is filled correctly

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
## ⚠️ Status

Actively developed.  
Stable for inference and RL workflows.

---

## 📜 License

MIT License


------------------------------------------------------------------------

## ⚡ Compute Requirements

-   Pretraining / finetuning: 4 GPUs\
-   Reinforcement Learning: 2 GPUs

------------------------------------------------------------------------


## 📜 License

MIT License
