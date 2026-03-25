from typing import Tuple

import torch
import torch.nn as nn


class PPOAgent(nn.Module):
    def __init__(
        self,
        cfg,
        adapter,
        stoi,
        model_checkpoint_path: str,
        trainable: bool,
        device: str,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.cfg = cfg
        self.adapter = adapter
        self.stoi = stoi
        self.trainable = trainable
        self.device_name = device
        self.dtype = dtype

        generation_cfg = cfg["generation"]
        self.generate_kwargs = {
            "seq_length": generation_cfg["seq_length"],
            "max_new_tokens": generation_cfg["max_new_tokens"],
            "top_k": generation_cfg["top_k"],
            "temperature": generation_cfg["temperature"],
            "eos_token": stoi["<EOS>"],
            "pad_token": stoi["<PAD>"],
        }

        GPTConfig, GPT = self.adapter.load_model_classes()

        checkpoint = torch.load(model_checkpoint_path, map_location=device)
        gptconf = GPTConfig(**checkpoint["model_args"])
        self.model = GPT(gptconf).to(device=device, dtype=dtype)

        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith(unwanted_prefix):
                k = k[len(unwanted_prefix) :]
            cleaned_state_dict[k] = v

        self.model.load_state_dict(cleaned_state_dict, strict=False)

        if not self.trainable:
            self.model.eval()
            self.model.requires_grad_(False)
        else:
            self.model.train()
            self.model.requires_grad_(True)
            self.adapter.maybe_freeze(self.model)

            n_embd = self.model.config.n_embd
            self.value_head = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.GELU(),
                nn.Linear(4 * n_embd, 1),
            ).to(device=device, dtype=dtype)

        self.logit_head = self.model.lm_head
        self.block_size = self.model.config.block_size
        self.n_embd = self.model.config.n_embd

    def _extra_model_kwargs(self, input_ids: torch.Tensor) -> dict:
        return self.adapter.build_model_kwargs(
            input_ids=input_ids,
            device=self.device_name,
            dtype=self.dtype,
            block_size=self.block_size,
            n_embd=self.n_embd,
        )

    def generate(self, input_ids: torch.Tensor, epoch: int) -> torch.Tensor:
        extra_kwargs = self._extra_model_kwargs(input_ids)
        return self.model.generate(
            idx=input_ids,
            epoch=epoch,
            **extra_kwargs,
            **self.generate_kwargs,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        extra_kwargs = self._extra_model_kwargs(input_ids)

        lm_logits, last_hidden_state = self.model(
            input_ids,
            attention_mask=attention_mask,
            return_hidden_states=True,
            **extra_kwargs,
        )

        if self.trainable:
            last_hidden_state = last_hidden_state.to(self.dtype)
            value = self.value_head(last_hidden_state).squeeze(-1)
            return lm_logits, value

        return lm_logits