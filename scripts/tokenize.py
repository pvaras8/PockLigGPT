import argparse
import os
from pathlib import Path

import yaml
from sklearn.model_selection import train_test_split

from pockliggpt.data.loaders.chembl import load_chembl_smiles
from pockliggpt.data.loaders.zinc20 import load_zinc_smiles
from pockliggpt.data.loaders.crossdocked import load_crossdocked

from pockliggpt.data.tokenizers.preprocessing import smiles_to_selfies
from pockliggpt.data.tokenizers.selfies_tokenizer import SelfiesTokenizer
from pockliggpt.data.tokenizers.pocket_ligand_tokenizer import PocketLigandTokenizer
from pockliggpt.data.tokenizers.io import save_to_bin_file, save_meta


def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize molecular datasets.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent_dir(path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def flatten_token_lists(token_lists):
    return [token for seq in token_lists for token in seq]


def tokenize_selfies_dataset(config: dict):
    dataset_cfg = config["dataset"]
    tokenizer_cfg = config["tokenizer"]
    output_cfg = config["output"]
    split_cfg = config["split"]

    dataset_type = dataset_cfg["type"]

    if dataset_type == "chembl":
        smiles = load_chembl_smiles(dataset_cfg["files"])
    elif dataset_type == "zinc20":
        smiles = load_zinc_smiles(dataset_cfg["files"])
    else:
        raise ValueError(f"Dataset type no soportado para SelfiesTokenizer: {dataset_type}")

    print(f"SMILES cargados: {len(smiles):,}")

    data = smiles_to_selfies(smiles)
    if data.empty:
        raise ValueError("No se pudieron convertir SMILES válidos a SELFIES.")

    print(f"SELFIES válidas: {len(data):,}")

    meta_path = tokenizer_cfg.get("meta_path")
    if meta_path:
        ensure_parent_dir(meta_path)

    tokenizer = SelfiesTokenizer(
    data=data,
    meta_path=meta_path if meta_path and os.path.exists(meta_path) else None,
    )

    if (
        meta_path
        and os.path.exists(meta_path)
        and tokenizer_cfg.get("extend_existing_vocab", False)
    ):
        tokenizer.extend_vocab(data["selfies"])
    max_length = int(tokenizer_cfg.get("max_length", 156))
    data = tokenizer.filter_valid_selfies(data, max_len=max_length)

    print(f"SELFIES tras filtrado por longitud: {len(data):,}")
    print(f"Vocab size: {tokenizer.vocab_size:,}")

    train_df, val_df = train_test_split(
        data,
        test_size=float(split_cfg.get("val_fraction", 0.1)),
        random_state=int(split_cfg.get("random_state", 42)),
    )

    train_ids = [tokenizer.tokenize(s, max_length=max_length) for s in train_df["selfies"]]
    val_ids = [tokenizer.tokenize(s, max_length=max_length) for s in val_df["selfies"]]

    output_dir = Path(output_cfg["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / output_cfg["train_file"]
    val_path = output_dir / output_cfg["val_file"]

    save_to_bin_file(flatten_token_lists(train_ids), train_path)
    save_to_bin_file(flatten_token_lists(val_ids), val_path)

    if meta_path:
        save_meta(tokenizer, meta_path)

    print(f"Train guardado en: {train_path}")
    print(f"Val guardado en:   {val_path}")
    if meta_path:
        print(f"Meta guardado en:  {meta_path}")


def tokenize_crossdocked_dataset(config: dict):
    dataset_cfg = config["dataset"]
    tokenizer_cfg = config["tokenizer"]
    output_cfg = config["output"]
    split_cfg = config["split"]

    pairs = load_crossdocked(dataset_cfg["file"])
    if pairs.empty:
        raise ValueError("No se encontraron pares pocket/smiles válidos.")

    print(f"Pares pocket/smiles cargados: {len(pairs):,}")

    data_sm = smiles_to_selfies(pairs["smiles"].tolist())
    if data_sm.empty:
        raise ValueError("No se pudieron convertir SMILES válidos a SELFIES en CrossDocked.")

    pairs = pairs.loc[data_sm.index].copy()
    pairs["selfies"] = data_sm["selfies"].values

    meta_path = tokenizer_cfg.get("meta_path")
    if meta_path:
        ensure_parent_dir(meta_path)

    tokenizer = PocketLigandTokenizer(
    data=pairs,
    meta_path=meta_path if meta_path and os.path.exists(meta_path) else None,
    )

    if (
        meta_path
        and os.path.exists(meta_path)
        and tokenizer_cfg.get("extend_existing_vocab", False)
    ):
        tokenizer.extend_vocab(pairs["selfies"])

    max_length = int(tokenizer_cfg.get("max_length", 156))
    pairs = tokenizer.filter_valid_pairs(pairs, max_sequence_length=max_length)

    print(f"Pares tras filtrado por longitud: {len(pairs):,}")
    print(f"Vocab size: {tokenizer.vocab_size:,}")

    train_df, val_df = train_test_split(
        pairs,
        test_size=float(split_cfg.get("val_fraction", 0.1)),
        random_state=int(split_cfg.get("random_state", 42)),
    )

    train_ids = [
        tokenizer.tokenize_pair(p, s, max_len=max_length)
        for p, s in zip(train_df["pocket"], train_df["selfies"])
    ]
    val_ids = [
        tokenizer.tokenize_pair(p, s, max_len=max_length)
        for p, s in zip(val_df["pocket"], val_df["selfies"])
    ]

    output_dir = Path(output_cfg["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / output_cfg["train_file"]
    val_path = output_dir / output_cfg["val_file"]

    save_to_bin_file(flatten_token_lists(train_ids), train_path)
    save_to_bin_file(flatten_token_lists(val_ids), val_path)

    if meta_path:
        save_meta(tokenizer, meta_path)

    print(f"Train guardado en: {train_path}")
    print(f"Val guardado en:   {val_path}")
    if meta_path:
        print(f"Meta guardado en:  {meta_path}")


def main():
    args = parse_args()
    config = load_config(args.config)

    dataset_type = config["dataset"]["type"]
    tokenizer_type = config["tokenizer"]["type"]

    print(f"Dataset type:   {dataset_type}")
    print(f"Tokenizer type: {tokenizer_type}")

    if dataset_type in {"chembl", "zinc20"} and tokenizer_type == "selfies":
        tokenize_selfies_dataset(config)

    elif dataset_type == "crossdocked" and tokenizer_type == "pocket_ligand":
        tokenize_crossdocked_dataset(config)

    else:
        raise ValueError(
            f"Combinación no soportada: dataset.type={dataset_type}, tokenizer.type={tokenizer_type}"
        )


if __name__ == "__main__":
    main()