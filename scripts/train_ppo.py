# scripts/train_ppo.py
import argparse
from omegaconf import OmegaConf

from pockliggpt.rl.model_adapters import build_model_adapter
from pockliggpt.rl.trainer import run_ppo_training


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    adapter = build_model_adapter(cfg)
    run_ppo_training(cfg, adapter)


if __name__ == "__main__":
    main()