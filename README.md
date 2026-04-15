# PockLigGPT

PockLigGPT is a pocket-conditioned molecular generation framework based on GPT architectures and reinforcement learning (RL) for structure-based drug design.

---

## 🌐 Online access

PockLigGPT is also available through a research collaboration interface:

👉 https://pockliggpt.streamlit.app

Researchers and industry partners can submit target proteins (PDB) and project descriptions to request molecule generation or full computational studies.

---

## 🔗 Model Weights

Download released checkpoints from Hugging Face:

👉 https://huggingface.co/pablovp8/PockLigGPT

Pretrained and fine-tuned checkpoints are provided for direct use.

---

## 🚀 Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## ⚡ Quickstart

```bash
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

---

### 3) Training

```bash
python scripts/train.py --config config/training/pretrain/zinc_20_sequence.yaml
python scripts/train.py --config config/training/finetune_1/chembl_sequence.yaml
python scripts/train.py --config config/training/finetune_2/crossdocked_sequence.yaml
```

---

### 4) Pretrained checkpoints

```bash
python -m huggingface_hub download pablovp8/PockLigGPT \
  --repo-type model \
  --local-dir checkpoints/pockliggpt
```

---

### 5) RL with pocket conditioning

```bash
python scripts/train_ppo.py --config config/rl/sequence_add.yaml
```

---

## 🧬 Pocket Embeddings

Use:

```bash
notebooks/prott5_pocket_pipeline_simple_en.ipynb
```

This generates:
- pocket amino-acid sequence
- ProtT5 residue embeddings (.npy)

---

## ⚙️ Docking setup

### Automatic download of docking_vina

```bash
bash scripts/setup_docking.sh
```

### MGLTools installation

Download from:
https://ccsb.scripps.edu/mgltools/

```bash
tar -zxvf mgltools_*.tar.gz
cd mgltools_x86_64Linux2_1.5.6
./install.sh
```

### Configure docking

Edit:

```bash
config/docking/vars_mgltools.json
```

This file contains installation-dependent paths that must be adapted to your local or HPC environment.

Review and update:

- nn1_script
- nn2_script
- prepare_ligand4.py
- prepare_receptor4.py
- mgl_python
- mgltools_directory
- filename_of_receptor
- root_output_folder
- source_compound_file
- output_directory
- final_folder

Also define the docking box:

- center_x, center_y, center_z
- size_x, size_y, size_z

---

## ✅ Minimal checklist

- datasets are in place
- tokenizer .pkl exists
- embeddings .npy generated
- checkpoint path valid
- docking_vina downloaded
- MGLTools installed
- docking config filled

---

## ⚡ Compute Requirements

- Pretraining / finetuning: 4 GPUs
- Reinforcement Learning: 2 GPUs

---

## 📜 License

MIT License
