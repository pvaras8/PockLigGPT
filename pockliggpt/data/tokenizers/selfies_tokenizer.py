import os
import pickle
import re
import pandas as pd


class SelfiesTokenizer:
    SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<LIGAND>", "</LIGAND>", "<UNK>"]

    def __init__(self, data=None, meta_path=None):
        if meta_path and os.path.exists(meta_path):
            self.load_meta(meta_path)
            self._ensure_special_tokens(self.SPECIAL_TOKENS)
            self.token2id = {v: k for k, v in self.id2token.items()}
        else:
            if data is None or "selfies" not in data:
                raise ValueError("Para crear un vocabulario nuevo necesitas data con columna 'selfies'.")
            self.id2token = self.build_tokenizer(data["selfies"])
            self.add_special_tokens()
            self.token2id = {v: k for k, v in self.id2token.items()}

    def split_selfies(self, selfies_str):
        if not isinstance(selfies_str, str):
            return []
        return re.findall(r"\[[^\]]+\]", selfies_str)

    def build_tokenizer(self, selfies_series):
        tokens = set()
        for s in selfies_series:
            if pd.notna(s):
                tokens.update(self.split_selfies(s))
        return {i: t for i, t in enumerate(sorted(tokens))}

    def add_special_tokens(self):
        idx = len(self.id2token)
        for t in self.SPECIAL_TOKENS:
            self.id2token[idx] = t
            idx += 1

    def _ensure_special_tokens(self, tokens):
        next_id = max(self.id2token) + 1 if self.id2token else 0
        existing = set(self.id2token.values())
        for t in tokens:
            if t not in existing:
                self.id2token[next_id] = t
                next_id += 1

    def extend_vocab(self, selfies_series):
        new_tokens = set()

        for s in selfies_series:
            if pd.notna(s):
                new_tokens.update(self.split_selfies(s))

        new_tokens = sorted(t for t in new_tokens if t not in self.token2id)

        if not new_tokens:
            print("No hay tokens nuevos que añadir al vocabulario.")
            return 0

        next_id = max(self.id2token) + 1
        for tok in new_tokens:
            self.id2token[next_id] = tok
            next_id += 1

        self.token2id = {v: k for k, v in self.id2token.items()}
        print(f"Añadidos {len(new_tokens)} tokens nuevos al vocabulario.")
        return len(new_tokens)

    def tokenize(self, selfies_str, max_length=156):
        toks = self.split_selfies(selfies_str)

        enc = [
            self.token2id["<SOS>"],
            self.token2id["<LIGAND>"],
        ]

        enc += [self.token2id.get(t, self.token2id["<UNK>"]) for t in toks]

        enc += [
            self.token2id["</LIGAND>"],
            self.token2id["<EOS>"],
        ]

        if len(enc) > max_length:
            raise ValueError(f"Sequence length {len(enc)} exceeds max_length={max_length}")

        if len(enc) < max_length:
            enc += [self.token2id["<PAD>"]] * (max_length - len(enc))

        return enc

    def decode(self, token_ids):
        skip = {"<PAD>", "<SOS>", "<EOS>", "<LIGAND>", "</LIGAND>", "<UNK>"}
        return "".join([self.id2token[i] for i in token_ids if self.id2token[i] not in skip])

    def filter_valid_selfies(self, df, max_len):
        return df[df["selfies"].apply(lambda x: len(self.split_selfies(x)) <= max_len)]

    def load_meta(self, path):
        with open(path, "rb") as f:
            meta = pickle.load(f)
        self.id2token = meta["itos"]
        self.token2id = meta["stoi"]

    @property
    def vocab_size(self):
        return len(self.id2token)