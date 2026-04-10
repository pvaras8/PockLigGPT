# PockLigGPT

PockLigGPT is a pocket-conditioned molecular generation framework based on GPT architectures and reinforcement learning (RL) for structure-based drug design.

It supports multiple workflows:

1. **Reproduce training** (pretraining + finetuning) from tokenized datasets.
2. **Use pretrained checkpoints** (e.g., from Hugging Face) and run RL.
3. **Condition RL with a real pocket** using sequence + ProtT5 embeddings (`.npy`) + receptor + docking coordinates.

---

## Installation

### Option A: `pip` (recommended for users)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

### Option B: `conda` (includes notebook tooling)

```bash
conda env create -f environment.yml
conda activate pockliggpt
python -m pip install -e .
```

Quick sanity check:

```bash
python -c "import pockliggpt; print(pockliggpt.__version__)"
```

---

## Repository workflow (end-to-end)

### 1) Prepare datasets

Raw data is **not included** in the repo due to size.

- See `datasets/README.md`
- Put your source files under `datasets/raw/...`

Typical sources used in this project:

- ChEMBL
- ZINC20
- CrossDocked

---

### 2) Tokenize datasets with the provided tokenizer + dictionary

Tokenization is done with `scripts/tokenize_dataset.py` and YAML configs in `config/tokenization/`.

Use the provided tokenizer dictionary (meta file):

- `datasets/tokenizer/meta_chembl_db_aa_2_proto4.pkl`

Run tokenization:

```bash
python scripts/tokenize_dataset.py --config config/tokenization/chembl.yaml
python scripts/tokenize_dataset.py --config config/tokenization/zinc20.yaml
python scripts/tokenize_dataset.py --config config/tokenization/crossdocked.yaml
```

This generates tokenized `.bin` files in `datasets/processed/...` (according to each config).

> Important: update input paths inside each YAML (`dataset.files` / `dataset.file`) to match your local machine.

---

### 3) Reproduce training (pretrain / finetune)

Training entrypoint:

- `scripts/train.py`

Config families:

- Pretrain: `config/training/pretrain/`
- Finetune stage 1: `config/training/finetune_1/`
- Finetune stage 2: `config/training/finetune_2/`

Examples:

```bash
python scripts/train.py --config config/training/pretrain/zinc_20_sequence.yaml
python scripts/train.py --config config/training/finetune_1/chembl_sequence.yaml
python scripts/train.py --config config/training/finetune_2/crossdocked_sequence.yaml
```

If using multi-GPU DDP, launch with `torchrun` and set `num_gpus`/devices in your config and environment.

---

### 4) Use pretrained checkpoints from Hugging Face (instead of full retraining)

You can skip long training by downloading released checkpoints and pointing your configs to them.

```bash
python -m huggingface_hub download <HF_USER_OR_ORG>/<HF_REPO> \
	--repo-type model \
	--local-dir checkpoints/pockliggpt
```

Then set:

- `model.checkpoint_path` in `config/rl/*.yaml`
- `training.init_ckpt_path` (if resuming finetuning)

---

### 5) RL with pocket conditioning (sequence + `.npy` + receptor + coordinates)

RL entrypoint:

```bash
python scripts/train_ppo.py --config config/rl/sequence_add.yaml
```

Alternative model configs:

- `config/rl/sequence.yaml`
- `config/rl/sequence_add.yaml`
- `config/rl/cross.yaml`

#### Step A — get pocket sequence and `.npy` from notebook

Use:

- `notebooks/prott5_pocket_pipeline_simple_en.ipynb`

From the notebook, export:

1. Pocket amino-acid sequence (space-separated, e.g. `ALA GLY ...`)
2. ProtT5 residue embeddings `.npy`

#### Step B — update RL config

In your selected `config/rl/*.yaml`, set:

- `conditioning.pocket_str` → your pocket sequence
- `conditioning.pocket_emb_path` → path to your `.npy`
- `model.checkpoint_path` → local checkpoint (trained or downloaded)
- `tokenizer.meta_path` → tokenizer dictionary path

#### Step C — set docking receptor and box coordinates

For docking-based reward, edit one vars file:

- `config/docking/vars_mgltools.json` (MGLTools pipeline), or
- `config/docking/vars_meeko.json` (Meeko/Vina pipeline)

At minimum define:

- receptor file path (`filename_of_receptor`)
- docking center (`center_x`, `center_y`, `center_z`)
- box size (`size_x`, `size_y`, `size_z`)

Then run PPO and the RL loop will generate molecules and optimize rewards automatically.

---

## Minimal checklist before running

- Dataset files downloaded and paths fixed in YAML
- Tokenizer `meta_chembl_db_aa_2_proto4.pkl` available
- RL config filled with sequence + `.npy`
- Docking vars JSON filled with receptor + coordinates
- Checkpoint path valid (local training output or Hugging Face download)

---

## Project highlights

- Pocket-conditioned generation
- SELFIES representation
- Multiple model adapters (`sequence`, `sequence_add`, `cross`)
- PPO-based RL training
- Docking reward integration

---

## Citation

If you use this code in your research, please cite the associated paper (to be added).
