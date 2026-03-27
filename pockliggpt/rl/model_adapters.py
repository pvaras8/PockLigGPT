import importlib
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .prompts import pocket_tokens_from_string


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"dtype no soportado: {dtype_name}")
    return mapping[dtype_name]


class BaseModelAdapter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_cfg = cfg["model"]
        self.prompt_cfg = cfg["prompt"]
        self.cond_cfg = cfg["conditioning"]
        self._pocket_template_cache: Dict[Tuple[str, str, int, int], torch.Tensor] = {}

    def load_model_classes(self):
        module = importlib.import_module(self.model_cfg["module"])
        return module.GPTConfig, module.GPT

    def build_prompt_prefix(self, stoi: Dict[str, int]):
        raise NotImplementedError

    def maybe_freeze(self, model) -> None:
        pass

    def build_model_kwargs(
        self,
        input_ids: torch.Tensor,
        device: str,
        dtype: torch.dtype,
        block_size: Optional[int] = None,
        n_embd: Optional[int] = None,
    ) -> dict:
        return {}

    def _require_conditioning_keys(self) -> None:
        required = ["pocket_str", "pocket_emb_path", "pocket_emb_aa_start"]
        model_type = self.model_cfg.get("type", "unknown")

        for key in required:
            if key not in self.cond_cfg:
                raise ValueError(
                    f"Falta conditioning.{key} para model.type={model_type}"
                )

    def _load_pocket_embedding_template(
        self,
        device: str,
        dtype: torch.dtype,
        block_size: int,
        n_embd: int,
    ) -> torch.Tensor:
        self._require_conditioning_keys()

        cache_key = (str(device), str(dtype), int(block_size), int(n_embd))
        if cache_key in self._pocket_template_cache:
            return self._pocket_template_cache[cache_key]

        pocket_str = self.cond_cfg["pocket_str"]
        pocket_emb_path = self.cond_cfg["pocket_emb_path"]
        aa_start = int(self.cond_cfg["pocket_emb_aa_start"])

        if aa_start < 0:
            raise ValueError("conditioning.pocket_emb_aa_start debe ser >= 0")

        pocket_emb_np = np.load(pocket_emb_path).astype(np.float32)

        aa_list = [aa.strip().upper() for aa in pocket_str.split()]
        l_seq = len(aa_list)

        if l_seq == 0:
            raise ValueError("conditioning.pocket_str está vacío")

        if pocket_emb_np.shape[0] == l_seq + 1:
            pocket_emb_np = pocket_emb_np[:l_seq]
        elif pocket_emb_np.shape[0] != l_seq:
            raise ValueError(
                f"Inconsistencia embeddings-pocket: {pocket_emb_np.shape[0]} vs {l_seq}"
            )

        num_res, d_prot = pocket_emb_np.shape
        if d_prot != n_embd:
            raise ValueError(
                f"Dim pocket ({d_prot}) != n_embd del modelo ({n_embd})"
            )

        if aa_start >= block_size:
            raise ValueError(
                f"conditioning.pocket_emb_aa_start={aa_start} fuera de block_size={block_size}"
            )

        pocket_emb_full = np.zeros((block_size, n_embd), dtype=np.float32)
        length_to_copy = min(num_res, block_size - aa_start)
        pocket_emb_full[aa_start : aa_start + length_to_copy, :] = pocket_emb_np[:length_to_copy, :]

        template = torch.from_numpy(pocket_emb_full).to(device=device, dtype=dtype)
        self._pocket_template_cache[cache_key] = template
        return template


class SequenceAdapter(BaseModelAdapter):
    def build_prompt_prefix(self, stoi: Dict[str, int]):
        include_pocket_tokens = self.prompt_cfg.get("include_pocket_tokens", True)
        if not include_pocket_tokens:
            raise ValueError(
                "SequenceAdapter espera prompt.include_pocket_tokens=true"
            )

        pocket_str = self.cond_cfg["pocket_str"]
        pocket_tokens = pocket_tokens_from_string(pocket_str, stoi)
        return [stoi["<SOS>"]] + pocket_tokens

    def build_model_kwargs(self, input_ids, device, dtype, block_size=None, n_embd=None):
        return {}


class SequenceAddAdapter(BaseModelAdapter):
    def build_prompt_prefix(self, stoi: Dict[str, int]):
        include_pocket_tokens = self.prompt_cfg.get("include_pocket_tokens", True)
        if not include_pocket_tokens:
            raise ValueError(
                "SequenceAddAdapter espera prompt.include_pocket_tokens=true"
            )

        pocket_str = self.cond_cfg["pocket_str"]
        pocket_tokens = pocket_tokens_from_string(pocket_str, stoi)
        return [stoi["<SOS>"]] + pocket_tokens

    def build_model_kwargs(self, input_ids, device, dtype, block_size=None, n_embd=None):
        if block_size is None or n_embd is None:
            raise ValueError("SequenceAddAdapter necesita block_size y n_embd")

        template = self._load_pocket_embedding_template(
            device=device,
            dtype=dtype,
            block_size=block_size,
            n_embd=n_embd,
        )
        bsz = input_ids.shape[0]
        pocket_emb = template.unsqueeze(0).expand(bsz, -1, -1)
        return {"pocket_emb": pocket_emb}


class CrossAdapter(BaseModelAdapter):
    def build_prompt_prefix(self, stoi: Dict[str, int]):
        include_pocket_tokens = self.prompt_cfg.get("include_pocket_tokens", False)
        if include_pocket_tokens:
            raise ValueError(
                "CrossAdapter espera prompt.include_pocket_tokens=false"
            )

        return [stoi["<SOS>"]]

    def maybe_freeze(self, model) -> None:
        freeze_cross = bool(self.model_cfg.get("freeze_cross_attention", False))
        if not freeze_cross:
            return

        # Reproduce la receta original PPO sobre cross-attention:
        # congelar módulos de cross_attn y su layer norm asociado.
        for name, p in model.named_parameters():
            if ("cross_attn" in name) or ("ln_cross" in name):
                p.requires_grad = False

    def build_model_kwargs(self, input_ids, device, dtype, block_size=None, n_embd=None):
        if block_size is None or n_embd is None:
            raise ValueError("CrossAdapter necesita block_size y n_embd")

        template = self._load_pocket_embedding_template(
            device=device,
            dtype=dtype,
            block_size=block_size,
            n_embd=n_embd,
        )
        bsz = input_ids.shape[0]
        pocket_emb = template.unsqueeze(0).expand(bsz, -1, -1)
        return {"pocket_emb": pocket_emb}


def build_model_adapter(cfg):
    model_type = cfg["model"]["type"]

    if model_type == "sequence":
        return SequenceAdapter(cfg)
    if model_type == "sequence_add":
        return SequenceAddAdapter(cfg)
    if model_type == "cross":
        return CrossAdapter(cfg)

    raise ValueError(f"model.type no soportado: {model_type}")