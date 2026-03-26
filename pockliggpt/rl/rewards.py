import os
import subprocess
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import selfies as sf
from numpy import exp
from rdkit import Chem
from rdkit.Chem import Descriptors


# =========================
# helpers químicos
# =========================

def calculate_logp(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return float(Descriptors.MolLogP(mol))
    return 0.0


def calculate_weight(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return float(Descriptors.MolWt(mol))
    return 0.0


# =========================
# funciones de reward
# =========================

def penalize_logP(logP, sigma=1.0):
    val = np.exp(-((logP - 2) ** 2) / (2 * sigma**2))
    if isinstance(logP, (int, float)):
        if logP < 0 or logP > 5:
            return 0.0
        return val
    val = np.where((logP < 0) | (logP > 5), 0.0, val)
    return val


def sigmoid_pen_docking(x, min_x=-12.0, max_x=-6.0, min_sig_in=-6, max_sig_in=6):
    sig = lambda y: 1 / (1 + exp(-y))
    input_map = lambda y: (y - min_x) / (max_x - min_x) * (max_sig_in - min_sig_in) + min_sig_in
    return sig(input_map(x))


def sigmoid_pen_pchembl(x, min_x=5.0, max_x=7.5, min_sig_in=-6, max_sig_in=6):
    sig = lambda y: 1 / (1 + exp(-y))
    input_map = lambda y: (y - min_x) / (max_x - min_x) * (max_sig_in - min_sig_in) + min_sig_in
    return sig(input_map(x))


def sigmoid_pen_reward(x, min_x=300.0, max_x=500, min_sig_in=-6, max_sig_in=6):
    sig = lambda y: 1 / (1 + exp(-y))
    input_map = lambda y: (y - min_x) / (max_x - min_x) * (max_sig_in - min_sig_in) + min_sig_in
    return sig(input_map(x))


def sigmoid_pen_delta(x, ref, min_x=-3.0, max_x=3.0, min_sig_in=-6, max_sig_in=6):
    delta = x - ref
    z = ((delta - min_x) / (max_x - min_x)) * (max_sig_in - min_sig_in) + min_sig_in
    return z


def combine_docking_only(d):
    return 1 - sigmoid_pen_docking(d)


def combine_docking_logP(docking, logp):
    def docking_component(d, d_min=-10.0, d_max=-4.0, s_min=-6.0, s_max=6.0, power=1.3):
        z = (d - d_min) / (d_max - d_min) * (s_max - s_min) + s_min
        base = 1.0 - (1.0 / (1.0 + np.exp(-z)))
        return np.clip(base, 0.0, 1.0) ** power

    def logp_component(lp, mu=2.5, sigma=1.0):
        return np.exp(-((lp - mu) ** 2) / (2.0 * sigma**2))

    d_comp = docking_component(docking)
    lp_comp = logp_component(logp)

    w_lp = 0.5
    bias = 0.2
    eps_floor = 0.02

    mod = (bias + w_lp * lp_comp) / (bias + w_lp)
    reward = np.clip(d_comp * mod, eps_floor, 1.0)
    return float(reward)


def combine_docking_logP_mw(docking, logp, mw):
    def sigmoid(x, center, width):
        return 1.0 / (1.0 + np.exp(-(x - center) / width))

    def docking_component(d, d_min=-12.0, d_max=-6.0, s_min=-6.0, s_max=6.0, power=1.3):
        z = (d - d_min) / (d_max - d_min) * (s_max - s_min) + s_min
        base = 1.0 - (1.0 / (1.0 + np.exp(-z)))
        return np.clip(base, 0, 1) ** power

    def logp_component(lp, mu=2.5, sigma=1.0):
        return np.exp(-((lp - mu) ** 2) / (2.0 * sigma**2))

    def mw_component(mw, center=500.0, width=60.0):
        sig = sigmoid(mw, center=center, width=width)
        return 1.0 - sig

    d_comp = docking_component(docking)
    lp_comp = logp_component(logp)
    mw_comp = mw_component(mw)

    w_lp, w_mw, bias = 0.4, 0.4, 0.3
    eps_floor = 0.02

    mod = (bias + w_lp * lp_comp + w_mw * mw_comp) / (bias + w_lp + w_mw)
    reward = np.clip(d_comp * mod, eps_floor, 1.0)
    return float(reward)


def combine_two_docking_specificity(d1, d2, k=25.0, tau=0.20, eps_floor=0.02):
    s1 = 1.0 - sigmoid_pen_docking(d1)
    s2 = 1.0 - sigmoid_pen_docking(d2)

    core = np.clip(s1 * (1.0 - s2), 0.0, 1.0)
    gate = 1.0 / (1.0 + np.exp(-k * ((s1 - s2) - tau)))

    r0 = np.clip(core * gate, 0.0, 1.0)
    r = eps_floor + (1.0 - eps_floor) * r0

    if isinstance(d1, (int, float)) and isinstance(d2, (int, float)):
        return float(r)
    return r


def combine_two_docking_specific_2(d1, d2, ref1, ref2, eps_floor=0.02):
    sig = lambda y: 1 / (1 + exp(-y))
    z1 = sigmoid_pen_delta(d1, ref1)
    z2 = sigmoid_pen_delta(d2, ref2)
    delta = z1 - z2

    r_spec = 1 - sig(delta)
    r_target = 1 - sigmoid_pen_docking(d1)

    r0 = r_spec * r_target
    r = eps_floor + (1 - eps_floor) * r0
    return float(r)


def combine_two_docking_specific_new(d1, d2, ref1, ref2, eps_floor=0.02):
    sig = lambda y: 1 / (1 + exp(-y))
    z1 = sigmoid_pen_delta(d1, ref1)
    z2 = sigmoid_pen_delta(d2, ref2)
    delta = z1 - z2

    gap = d2 - d1
    tau_gap = 1.0

    if (d2 < ref2) and (gap < tau_gap):
        return float(eps_floor)

    r_spec = 1 - sig(delta)
    r_target = 1 - sigmoid_pen_docking(d1)

    r0 = r_spec * r_target
    r = eps_floor + (1 - eps_floor) * r0
    return float(r)


COMBINERS: Dict[str, Callable] = {
    "docking_only": combine_docking_only,
    "docking_logp": combine_docking_logP,
    "docking_logp_mw": combine_docking_logP_mw,
    "two_docking_specificity": combine_two_docking_specificity,
    "two_docking_specific_2": combine_two_docking_specific_2,
    "two_docking_specific_new": combine_two_docking_specific_new,
     "mw_only": sigmoid_pen_reward,  # 👈 NUEVO

}


class RewardRunner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reward_cfg = cfg["reward"]

        script = self.reward_cfg.get("script", None)
        vars_file = self.reward_cfg.get("vars_file", None)

        self.reward_script = os.path.abspath(script) if script else None
        self.vars_file = os.path.abspath(vars_file) if vars_file else None
        self.smiles_output_file = os.path.abspath(self.reward_cfg["smiles_output_file"])
        self.output_prefix = self.reward_cfg.get("output_prefix", "reward_results")
        self.combiner_name = self.reward_cfg["combiner"]

        if self.combiner_name not in COMBINERS:
            raise ValueError(f"Combiner desconocido: {self.combiner_name}")

        self.combiner = COMBINERS[self.combiner_name]

    def _decode_selfies(self, molecules_selfies: List[str]) -> List[str]:
        molecules_smiles = []
        for selfie in molecules_selfies:
            try:
                smiles = sf.decoder(selfie)
                molecules_smiles.append(smiles)
            except Exception:
                molecules_smiles.append("C")
        return molecules_smiles

    def _prepare_smiles_file(self, smiles_list: List[str]) -> None:
        with open(self.smiles_output_file, "w") as f:
            for smi in smiles_list:
                f.write(smi + "\n")

    def _apply_combiner(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.combiner_name == "docking_only":
            df["Fitness"] = df["Docking"].apply(self.combiner)

        elif self.combiner_name == "docking_logp":
            df["LogP"] = df["SMILES"].apply(calculate_logp)
            df["Fitness"] = df.apply(lambda row: self.combiner(row["Docking"], row["LogP"]), axis=1)

        elif self.combiner_name == "docking_logp_mw":
            df["LogP"] = df["SMILES"].apply(calculate_logp)
            df["MW"] = df["SMILES"].apply(calculate_weight)
            df["Fitness"] = df.apply(
                lambda row: self.combiner(row["Docking"], row["LogP"], row["MW"]),
                axis=1,
            )
        elif self.combiner_name == "mw_only":
            df["MW"] = df["SMILES"].apply(calculate_weight)
            df["Fitness"] = df["MW"].apply(self.combiner)

        else:
            raise NotImplementedError(
                f"Combiner '{self.combiner_name}' requiere columnas extra; amplíalo aquí."
            )

        return df

    def __call__(self, molecules_selfies: List[str], epoch: int) -> List[float]:
        molecules_smiles = self._decode_selfies(molecules_selfies)
        valid_molecules = [m for m in molecules_smiles if m is not None]

        if len(valid_molecules) == 0:
            return [0.0] * len(molecules_selfies)

        self._prepare_smiles_file(valid_molecules)

        docking_output_file_temp = os.path.join(
            os.path.dirname(self.smiles_output_file),
            f"{self.output_prefix}_{epoch}_temp.csv",
        )
        global_docking_output_file = os.path.join(
            os.path.dirname(self.smiles_output_file),
            f"{self.output_prefix}_{epoch}.csv",
        )

        try:
            if self.reward_script is not None:
                print(f"Llamando a {self.reward_script} para docking en época {epoch}...")
                subprocess.run(
                    ["python", self.reward_script, self.smiles_output_file, self.vars_file, str(epoch)],
                    check=True,
                )

                df_temp = pd.read_csv(docking_output_file_temp)

                if "Docking" not in df_temp.columns:
                    raise ValueError("El CSV de docking no contiene columna 'Docking'.")

            else:
                # 👇 caso sin script (ej: MW, LogP, etc.)
                df_temp = pd.DataFrame({"SMILES": valid_molecules})

            df_temp = self._apply_combiner(df_temp)

            if os.path.exists(global_docking_output_file):
                df_global = pd.read_csv(global_docking_output_file)
                df_final = pd.concat([df_global, df_temp], ignore_index=True)
            else:
                df_final = df_temp

            df_final.to_csv(global_docking_output_file, index=False)

            return df_temp["Fitness"].tolist()

        except Exception as e:
            print(f"Error durante reward_fn: {e}")
            return [0.0] * len(valid_molecules)