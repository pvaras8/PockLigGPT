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

Typical datasets: - ChEMBL\
- ZINC20\
- CrossDocked

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

## ⚙️ Docking Setup

``` bash
bash scripts/setup_docking.sh
```

Requires MGLTools.

------------------------------------------------------------------------

## ⚡ Compute Requirements

-   Pretraining / finetuning: 4 GPUs\
-   Reinforcement Learning: 2 GPUs

------------------------------------------------------------------------

## 💼 Contact

👉 https://pockliggpt.streamlit.app

------------------------------------------------------------------------

## 📜 License

MIT License
