import os
import pickle
import re


AA_ONE_TO_TOKEN = {
    "A": "<AA_ALA>", "C": "<AA_CYS>", "D": "<AA_ASP>", "E": "<AA_GLU>",
    "F": "<AA_PHE>", "G": "<AA_GLY>", "H": "<AA_HIS>", "I": "<AA_ILE>",
    "K": "<AA_LYS>", "L": "<AA_LEU>", "M": "<AA_MET>", "N": "<AA_ASN>",
    "P": "<AA_PRO>", "Q": "<AA_GLN>", "R": "<AA_ARG>", "S": "<AA_SER>",
    "T": "<AA_THR>", "V": "<AA_VAL>", "W": "<AA_TRP>", "Y": "<AA_TYR>",
}


class PocketLigandTokenizer:
    SPECIAL_TOKENS = [
        "<PAD>", "<SOS>", "<EOS>", "<UNK>",
        "<LIGAND>", "</LIGAND>", "<POCKET>", "</POCKET>"
    ]

    def __init__(self, data=None, meta_path=None):
        if meta_path and os.path.exists(meta_path):
            self.load_meta(meta_path)
            self._ensure_special_tokens(self.SPECIAL_TOKENS + list(AA_ONE_TO_TOKEN.values()))
        else:
            if data is None or "selfies" not in data:
                raise ValueError("Para crear un vocabulario nuevo necesitas data con columna 'selfies'.")
            self.id2token = self.build_tokenizer(data["selfies"])
            self.add_special_tokens()
            self._ensure_special_tokens(list(AA_ONE_TO_TOKEN.values()))

        self.token2id = {v: k for k, v in self.id2token.items()}

    def split_selfies(self, s):
        if not isinstance(s, str):
            return []
        return re.findall(r"\[[^\]]+\]", s)

    def build_tokenizer(self, selfies_series):
        tokens = set()
        for s in selfies_series:
            if isinstance(s, str):
                tokens.update(self.split_selfies(s))
        return {i: t for i, t in enumerate(sorted(tokens))}

    def add_special_tokens(self):
        base = len(self.id2token)
        for i, t in enumerate(self.SPECIAL_TOKENS, start=base):
            self.id2token[i] = t

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
            if isinstance(s, str):
                new_tokens.update(self.split_selfies(s))

        new_tokens = sorted(t for t in new_tokens if t not in self.token2id)

        if not new_tokens:
            print("No hay tokens SELFIES nuevos que añadir al vocabulario.")
            return 0

        next_id = max(self.id2token) + 1
        for tok in new_tokens:
            self.id2token[next_id] = tok
            next_id += 1

        self.token2id = {v: k for k, v in self.id2token.items()}
        print(f"Añadidos {len(new_tokens)} tokens SELFIES nuevos al vocabulario.")
        return len(new_tokens)

    def core_length(self, pocket_str, selfies_str):
        return len(pocket_str) + len(self.split_selfies(selfies_str))

    def filter_valid_pairs(self, df_pairs, max_sequence_length):
        max_core = max_sequence_length - 6
        keep = []
        for _, row in df_pairs.iterrows():
            keep.append(self.core_length(row["pocket"], row["selfies"]) <= max_core)
        return df_pairs[keep]

    def tokenize_pair(self, pocket, selfies, max_len=156):
        seq = ["<SOS>", "<POCKET>"]
        seq += [AA_ONE_TO_TOKEN.get(a, "<UNK>") for a in pocket]
        seq += ["</POCKET>", "<LIGAND>"]
        seq += self.split_selfies(selfies)
        seq += ["</LIGAND>", "<EOS>"]

        ids = [self.token2id.get(t, self.token2id["<UNK>"]) for t in seq]

        if len(ids) > max_len:
            raise ValueError(f"Sequence length {len(ids)} exceeds max_len={max_len}")

        if len(ids) < max_len:
            ids += [self.token2id["<PAD>"]] * (max_len - len(ids))

        return ids

    def load_meta(self, path):
        with open(path, "rb") as f:
            meta = pickle.load(f)
        self.id2token = meta["itos"]
        self.token2id = meta["stoi"]

    @property
    def vocab_size(self):
        return len(self.id2token)